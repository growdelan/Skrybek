# main.py
# macOS (Apple Silicon) | Python 3.13+
# Flow: record mic -> stop after N sec silence OR ESC -> save WAV -> transcribe (mlx-whisper)
# -> (optional) post-process text with LiteRT (gemma4 E2B) -> print + copy to clipboard.

from __future__ import annotations

import argparse
import contextlib
import queue
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

SYSTEM_PROMPT = (
    "Jesteś lokalnym modułem dyktowania dla aplikacji terminalowej. "
    "Użytkownik mówi po polsku, a Twoim zadaniem jest zwrócić wyłącznie "
    "wierny zapis tego, co powiedział, zawsze w języku polskim. "
    "Masz zachować 100 procent znaczenia i treści wypowiedzi. "
    "Nie wolno Ci parafrazować, skracać, rozwijać, dopowiadać, tłumaczyć "
    "ani zmieniać doboru słów bardziej, niż wymaga tego zapis pisemny. "
    "Wolno Ci poprawić tylko interpunkcję, wielkie litery, oczywiste rozdzielenie zdań "
    "oraz minimalne wygładzenie zapisu konieczne do czytelności, bez zmiany sensu. "
    "Nie dodawaj żadnych komentarzy, wstępów, etykiet ani wyjaśnień. "
    "Nie zwracaj surowej transkrypcji. "
    "Zawsze użyj narzędzia return_final_text."
)

CONVERSATIONAL_PREFIX_RE = re.compile(
    r"^\s*(oto|poniżej|jasne|pewnie|oczywiście|transkrypcja|poprawiona transkrypcja)\b",
    re.IGNORECASE,
)
CONTROL_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")
TOOL_WRAPPER_RE = re.compile(
    r"^\s*return_final_text\s*\{\s*final_text\s*:\s*(.*?)\s*\}\s*[.!?]?\s*$",
    re.IGNORECASE | re.DOTALL,
)

DEFAULT_OUTPUT_WAV = "recording.wav"
CYAN_PROMPT = "\033[96m"
ANSI_RESET = "\033[0m"
DEFAULT_WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"

_PARAKEET_MODEL_CACHE: dict[str, Any] = {}


class LiteRTError(Exception):
    """Bazowy błąd obróbki LiteRT."""


class LiteRTUnavailableError(LiteRTError):
    """Model LiteRT jest niedostępny."""


class LiteRTProcessingError(LiteRTError):
    """LiteRT nie zwrócił poprawnego wyniku."""


class STTError(Exception):
    """Bazowy błąd backendu rozpoznawania mowy."""


class STTUnavailableError(STTError):
    """Backend STT lub jego zależności są niedostępne."""


class STTProcessingError(STTError):
    """Backend STT nie zwrócił poprawnego wyniku."""


def resolve_litert_system_prompt(prompt_input: Optional[str]) -> str:
    """
    Rozstrzyga źródło promptu LiteRT:
    - brak flagi: wbudowany SYSTEM_PROMPT,
    - istniejący plik: zawartość pliku,
    - w przeciwnym razie: tekst inline z flagi.
    """
    if prompt_input is None:
        return SYSTEM_PROMPT

    candidate = Path(prompt_input).expanduser()
    if candidate.exists():
        if not candidate.is_file():
            raise LiteRTUnavailableError(
                f"Ścieżka promptu nie jest plikiem: {candidate}"
            )
        try:
            file_text = candidate.read_text(encoding="utf-8").strip()
        except Exception as exc:
            raise LiteRTUnavailableError(
                f"Nie udało się odczytać pliku promptu: {candidate}"
            ) from exc
        if not file_text:
            raise LiteRTUnavailableError(f"Plik promptu jest pusty: {candidate}")
        return file_text

    inline_text = prompt_input.strip()
    if not inline_text:
        raise LiteRTUnavailableError("Prompt podany flagą --prompt jest pusty.")
    return inline_text


class LiteRTTextProcessor:
    """Minimalny klient LiteRT-LM do korekty transkrypcji tekstowej."""

    def __init__(self, model_path: str, verbose: bool = False):
        if not model_path:
            raise LiteRTUnavailableError(
                "Brak ścieżki modelu LiteRT. Użyj --litert-model-path."
            )

        path = Path(model_path).expanduser()
        if not path.exists() or not path.is_file():
            raise LiteRTUnavailableError(f"Model nie istnieje pod ścieżką: {path}")

        try:
            import litert_lm
        except ImportError as exc:
            raise LiteRTUnavailableError("Brak zależności litert-lm w środowisku.") from exc

        self._engine = None
        self._verbose = verbose

        try:
            if verbose:
                print(f"[litert] loading model: {path}", flush=True)
            self._engine = litert_lm.Engine(
                str(path),
                backend=litert_lm.Backend.GPU,
                audio_backend=litert_lm.Backend.CPU,
            )
            self._engine.__enter__()
        except Exception as exc:
            raise LiteRTUnavailableError(
                f"Nie udało się załadować modelu LiteRT-LM: {exc}"
            ) from exc

    def close(self) -> None:
        if self._engine is not None:
            self._engine.__exit__(None, None, None)
            self._engine = None

    def process_text(self, transcript: str, system_prompt_text: str) -> str:
        if not transcript.strip():
            return ""

        if self._engine is None:
            raise LiteRTUnavailableError("Model LiteRT nie został poprawnie zainicjalizowany.")

        tool_result: dict[str, str] = {}

        def return_final_text(final_text: str) -> str:
            tool_result["text"] = final_text
            return "OK"

        content = [
            {
                "type": "text",
                "text": (
                    "TRANSKRYPCJA UŻYTKOWNIKA:\n"
                    f"{transcript}"
                ),
            }
        ]

        try:
            with self._engine.create_conversation(
                messages=[{"role": "system", "content": system_prompt_text}],
                tools=[return_final_text],
            ) as conversation:
                response = conversation.send_message({"role": "user", "content": content})
        except Exception as exc:
            raise LiteRTProcessingError(f"LiteRT-LM nie przetworzył tekstu: {exc}") from exc

        if tool_result.get("text"):
            return normalize_final_text(tool_result["text"])

        try:
            fallback = response["content"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LiteRTProcessingError("Model nie zwrócił rozpoznawalnej odpowiedzi.") from exc

        return normalize_final_text(fallback)


def process_text_with_litert_subprocess(
    transcript: str,
    model_path: str,
    prompt_input: Optional[str],
    verbose: bool = False,
) -> str:
    """
    Uruchamia LiteRT w osobnym procesie, aby uniknąć konfliktów runtime/GPU
    z innymi backendami STT działającymi w tym samym interpreterze.
    """
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_litert-subprocess",
        "--litert-model-path",
        model_path,
    ]
    if prompt_input is not None:
        cmd.extend(["--prompt", prompt_input])
    if verbose:
        cmd.append("--verbose")

    try:
        completed = subprocess.run(
            cmd,
            input=transcript,
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception as exc:
        raise LiteRTUnavailableError(
            f"Nie udało się uruchomić procesu LiteRT: {exc}"
        ) from exc

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        details = stderr or stdout or "LiteRT helper zakończył się błędem."
        raise LiteRTProcessingError(details)

    output = completed.stdout.strip()
    if not output:
        raise LiteRTProcessingError("LiteRT helper nie zwrócił tekstu.")
    vprint(verbose, "[litert] processed in subprocess")
    return output


def vprint(verbose: bool, *args, **kwargs):
    if verbose:
        print(*args, **kwargs, flush=True)


def normalize_final_text(text: str) -> str:
    """Czyści wynik modelu i odrzuca odpowiedzi konwersacyjne."""
    normalized = CONTROL_TOKEN_RE.sub(" ", text)
    wrapper_match = TOOL_WRAPPER_RE.match(normalized)
    if wrapper_match:
        normalized = wrapper_match.group(1)
    normalized = " ".join(normalized.strip().split())
    normalized = normalized.strip("\"' \n\t")
    if not normalized:
        raise LiteRTProcessingError("Model nie zwrócił finalnego tekstu.")
    if CONVERSATIONAL_PREFIX_RE.match(normalized):
        raise LiteRTProcessingError(
            "Model zwrócił odpowiedź konwersacyjną zamiast finalnego tekstu."
        )
    if normalized[-1] not in ".!?":
        normalized += "."
    if normalized[0].isalpha():
        normalized = normalized[0].upper() + normalized[1:]
    return normalized


# --- Nagrywanie do wykrycia ciszy lub ESC ---
def record_until_silence(
    rate: int = 16000,
    silence_threshold: float = 0.01,
    silence_seconds: float = 3.0,
    device: Optional[int | str] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Nagrywa mono float32 [-1, 1] do czasu:
      - wykrycia ciągłej ciszy przez silence_seconds (RMS < silence_threshold) LUB
      - naciśnięcia klawisza ESC w terminalu (tty).
    """
    sd.default.samplerate = rate
    sd.default.channels = 1
    if device is not None:
        sd.default.device = device

    blocksize = max(512, int(rate * 0.064))  # ~64 ms
    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
    collected: List[np.ndarray] = []

    silent_accum = 0.0
    started = False

    def callback(indata, frames, time_info, status):
        if status:
            pass
        audio_q.put(indata.copy())

    # Przygotuj nieblokujący odczyt klawiatury (tylko POSIX + TTY)
    use_kb = sys.stdin.isatty()
    kb_fd = None
    kb_state = None
    if use_kb:
        try:
            import termios, tty  # tylko na POSIX (macOS)

            kb_fd = sys.stdin.fileno()
            kb_state = termios.tcgetattr(kb_fd)
            tty.setcbreak(kb_fd)  # nie blokuje i nie czeka na Enter
            vprint(verbose, "[rec] Press ESC to stop recording manually.")
        except Exception:
            use_kb = False
            kb_fd = None
            kb_state = None

    vprint(
        verbose,
        f"[rec] start (rate={rate}, blocksize={blocksize}, threshold={silence_threshold}, silence={silence_seconds}s) …",
    )

    try:
        import select  # POSIX

        with sd.InputStream(callback=callback, blocksize=blocksize, dtype="float32"):
            last_print = time.time()
            while True:
                # 0) Sprawdź klawisz ESC (nieblokująco)
                if use_kb:
                    try:
                        r, _, _ = select.select([sys.stdin], [], [], 0)
                        if r:
                            ch = sys.stdin.read(1)
                            if ch == "\x1b":  # ESC
                                vprint(verbose, "[rec] ESC pressed -> stopping")
                                break
                    except Exception:
                        # Jeśli coś pójdzie nie tak, po prostu ignoruj klawiaturę
                        use_kb = False

                # 1) Odbierz audio (krótki timeout, żeby pętla była responsywna)
                try:
                    chunk = audio_q.get(timeout=0.1)
                except queue.Empty:
                    # brak nowego audio, pętla i tak sprawdza ESC często
                    continue

                collected.append(chunk)
                rms = float(np.sqrt(np.mean(np.square(chunk.astype(np.float32)))))
                dur = chunk.shape[0] / rate

                if rms >= silence_threshold:
                    started = True
                    silent_accum = 0.0
                else:
                    silent_accum += dur

                # 2) Warunek zatrzymania: ciągła cisza
                if started and silent_accum >= silence_seconds:
                    vprint(
                        verbose,
                        f"[rec] detected {silence_seconds:.2f}s silence -> stopping",
                    )
                    break

                # 3) Log co ~1 s
                now = time.time()
                if verbose and now - last_print > 1.0:
                    total_dur = sum(c.shape[0] for c in collected) / rate
                    print(
                        f"[rec] dur={total_dur:.1f}s, rms={rms:.4f}, silent_accum={silent_accum:.2f}s",
                        flush=True,
                    )
                    last_print = now
    except KeyboardInterrupt:
        vprint(verbose, "[rec] Interrupted by user (Ctrl+C) -> stopping")
    except Exception as e:
        print(f"[rec][ERR] {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Przywróć tryb terminala
        if use_kb and kb_fd is not None and kb_state is not None:
            try:
                import termios

                termios.tcsetattr(kb_fd, termios.TCSADRAIN, kb_state)
            except Exception:
                pass

    audio = (
        np.concatenate(collected, axis=0).reshape(-1)
        if collected
        else np.zeros(0, dtype=np.float32)
    )
    total = audio.shape[0] / rate
    vprint(verbose, f"[rec] captured {total:.2f}s, samples={audio.shape[0]}")
    return audio


# --- Zapis WAV (PCM16, mono, 16 kHz) ---
def save_wav_pcm16(
    path: str, rate: int, audio_f32_mono: np.ndarray, verbose: bool = False
) -> None:
    if audio_f32_mono.ndim != 1:
        audio_f32_mono = audio_f32_mono.reshape(-1)
    audio_i16 = np.clip(audio_f32_mono, -1.0, 1.0)
    audio_i16 = (audio_i16 * 32767.0).astype(np.int16)
    wavfile.write(path, rate, audio_i16)
    vprint(verbose, f"[wav] saved -> {path} ({len(audio_i16) / rate:.2f}s)")


# --- Transkrypcja (mlx-whisper) ---
def transcribe_with_whisper(
    wav_path: str,
    whisper_model: Optional[str] = None,
    verbose: bool = False,
) -> str:
    """
    Używa oficjalnego API: mlx_whisper.transcribe(file, path_or_hf_repo=...) -> dict['text'].
    """
    try:
        import mlx_whisper  # PyPI: mlx-whisper
    except ImportError as exc:
        raise STTUnavailableError(
            "Brak zależności mlx-whisper w środowisku. Zainstaluj projekt ponownie."
        ) from exc

    kwargs = {}
    if whisper_model:
        kwargs["path_or_hf_repo"] = whisper_model
    kwargs["language"] = "pl"

    vprint(
        verbose,
        f"[whisper] transcribing {wav_path} with model={whisper_model or 'default'} …",
    )
    try:
        out = mlx_whisper.transcribe(wav_path, **kwargs)
    except Exception as exc:
        raise STTProcessingError(f"mlx-whisper nie przetworzył pliku audio: {exc}") from exc
    text = (out.get("text") or "").strip()
    segments = out.get("segments") or []
    if not text:
        raise STTProcessingError("mlx-whisper zwrócił pustą transkrypcję.")

    seg_info = f", segments={len(segments)}" if segments else ""
    vprint(verbose, f"[whisper] transcript len={len(text)} chars{seg_info}")
    return text


def _load_parakeet_model(model_name: str, verbose: bool = False) -> Any:
    if model_name in _PARAKEET_MODEL_CACHE:
        vprint(verbose, f"[parakeet] reusing cached model: {model_name}")
        return _PARAKEET_MODEL_CACHE[model_name]

    try:
        from mlx_audio.stt.utils import load
    except ImportError:
        try:
            from mlx_audio.stt import load  # type: ignore[attr-defined]
        except ImportError as exc:
            raise STTUnavailableError(
                "Brak zależności mlx-audio w środowisku. "
                "Zainstaluj projekt z nowymi zależnościami, aby użyć backendu parakeet."
            ) from exc

    try:
        vprint(verbose, f"[parakeet] loading model: {model_name}")
        model = load(model_name)
    except Exception as exc:
        raise STTUnavailableError(
            f"Nie udało się załadować modelu Parakeet: {exc}"
        ) from exc

    _PARAKEET_MODEL_CACHE[model_name] = model
    return model


def transcribe_with_parakeet(
    wav_path: str,
    parakeet_model: str = DEFAULT_PARAKEET_MODEL,
    verbose: bool = False,
) -> str:
    """
    Używa mlx-audio do uruchomienia modelu Parakeet w formacie MLX.
    """
    model = _load_parakeet_model(parakeet_model, verbose=verbose)
    vprint(verbose, f"[parakeet] transcribing {wav_path} with model={parakeet_model} …")

    try:
        result = model.generate(wav_path)
    except Exception as exc:
        raise STTProcessingError(f"Parakeet nie przetworzył pliku audio: {exc}") from exc

    text = getattr(result, "text", "")
    if not isinstance(text, str):
        text = str(text or "")
    text = text.strip()
    if not text:
        raise STTProcessingError("Parakeet zwrócił pustą transkrypcję.")

    sentence_count = len(getattr(result, "sentences", []) or [])
    sent_info = f", sentences={sentence_count}" if sentence_count else ""
    vprint(verbose, f"[parakeet] transcript len={len(text)} chars{sent_info}")
    return text


def transcribe_audio(
    wav_path: str,
    stt_backend: str,
    whisper_model: str,
    parakeet_model: str,
    verbose: bool = False,
) -> str:
    if stt_backend == "whisper":
        return transcribe_with_whisper(
            wav_path,
            whisper_model=whisper_model,
            verbose=verbose,
        )
    if stt_backend == "parakeet":
        return transcribe_with_parakeet(
            wav_path,
            parakeet_model=parakeet_model,
            verbose=verbose,
        )
    raise STTUnavailableError(f"Nieznany backend STT: {stt_backend}")


# --- Schowek ---
def copy_to_clipboard(text: str, verbose: bool = False) -> bool:
    try:
        import pyperclip

        pyperclip.copy(text)
        vprint(verbose, "[clip] copied via pyperclip")
        return True
    except Exception as e:
        vprint(verbose, f"[clip] pyperclip failed: {e}; trying pbcopy")
        try:
            p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
            p.communicate(input=text.encode("utf-8"))
            ok = p.returncode == 0
            if ok:
                vprint(verbose, "[clip] copied via pbcopy")
            else:
                vprint(verbose, "[clip] pbcopy failed")
            return ok
        except Exception as e2:
            vprint(verbose, f"[clip] pbcopy error: {e2}")
            return False


def ask_continue_recording() -> bool:
    """Pytanie o kontynuację nagrywania. Zwraca True dla T, False dla N."""
    while True:
        answer = input(
            f"{CYAN_PROMPT}Czy nagrać kolejną wiadomość [T/N]: {ANSI_RESET}"
        ).strip()
        normalized = answer.lower()
        if normalized == "t":
            return True
        if normalized == "n":
            return False
        print("[warn] Niepoprawna odpowiedź. Wpisz T lub N.")


# --- CLI / main ---
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Record -> Transcribe (Whisper or Parakeet) -> (optional) Process (LiteRT) -> Print + Clipboard"
    )
    ap.add_argument(
        "--stt-backend",
        type=str,
        choices=("whisper", "parakeet"),
        default="whisper",
        help="Speech-to-text backend: whisper (mlx-whisper) or parakeet (mlx-audio).",
    )
    ap.add_argument(
        "--use-litert",
        action="store_true",
        help="If set, post-process transcript with LiteRT; otherwise output raw transcript.",
    )
    ap.add_argument(
        "--litert-model-path",
        type=str,
        default=None,
        help="Path to local LiteRT model file (.litertlm), required with --use-litert.",
    )
    ap.add_argument(
        "--rate", type=int, default=16000, help="Sample rate (Hz), default 16000"
    )
    ap.add_argument(
        "--silence-threshold",
        type=float,
        default=0.01,
        help="RMS threshold for silence (0..1), default 0.01",
    )
    ap.add_argument(
        "--silence-seconds",
        type=float,
        default=3.0,
        help="Continuous silence to stop (s), default 3.0",
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional sounddevice input device id/name",
    )
    ap.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="System prompt override for LiteRT: inline text or path to a prompt file (used only with --use-litert).",
    )
    ap.add_argument(
        "--whisper-model",
        type=str,
        default=DEFAULT_WHISPER_MODEL,
        help="Whisper model id/path for mlx-whisper (e.g., mlx-community/whisper-large-v3-turbo)",
    )
    ap.add_argument(
        "--parakeet-model",
        type=str,
        default=DEFAULT_PARAKEET_MODEL,
        help="Parakeet model id/path for mlx-audio (e.g., mlx-community/parakeet-tdt-0.6b-v3)",
    )
    ap.add_argument(
        "--_litert-subprocess",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    ap.add_argument("--verbose", action="store_true", help="Verbose logs")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    verbose = args.verbose

    if args._litert_subprocess:
        transcript = sys.stdin.read()
        if not args.litert_model_path:
            print(
                "[litert][ERR] Brak --litert-model-path. Podaj ścieżkę do lokalnego pliku .litertlm.",
                file=sys.stderr,
            )
            return 2

        processor: LiteRTTextProcessor | None = None
        try:
            system_prompt = resolve_litert_system_prompt(args.prompt)
            with contextlib.redirect_stdout(sys.stderr):
                processor = LiteRTTextProcessor(args.litert_model_path, verbose=verbose)
                final_text = processor.process_text(
                    transcript=transcript,
                    system_prompt_text=system_prompt,
                )
        except LiteRTError as exc:
            print(f"[litert][ERR] {exc}", file=sys.stderr)
            return 1
        finally:
            if processor is not None:
                processor.close()

        print(final_text)
        return 0

    while True:
        # 1) Record
        audio = record_until_silence(
            rate=args.rate,
            silence_threshold=args.silence_threshold,
            silence_seconds=args.silence_seconds,
            device=args.device,
            verbose=verbose,
        )

        if audio.size == 0:
            print("[ERR] No audio captured.", file=sys.stderr)
            return 2

        # 2) Save WAV (PCM16)
        save_wav_pcm16(DEFAULT_OUTPUT_WAV, args.rate, audio, verbose=verbose)

        # 3) Transcribe locally
        try:
            transcript = transcribe_audio(
                DEFAULT_OUTPUT_WAV,
                stt_backend=args.stt_backend,
                whisper_model=args.whisper_model,
                parakeet_model=args.parakeet_model,
                verbose=verbose,
            )
        except STTError as exc:
            print(f"[stt][ERR] {exc}", file=sys.stderr)
            return 1

        # 4) (Optional) Process text with LiteRT
        if args.use_litert:
            if not args.litert_model_path:
                print(
                    "[ERR] Brak --litert-model-path. Podaj ścieżkę do lokalnego pliku .litertlm.",
                    file=sys.stderr,
                )
                return 2

            try:
                final_text = process_text_with_litert_subprocess(
                    transcript=transcript,
                    model_path=args.litert_model_path,
                    prompt_input=args.prompt,
                    verbose=verbose,
                )
            except LiteRTError as exc:
                print(f"[litert][ERR] {exc}", file=sys.stderr)
                return 1
        else:
            vprint(verbose, "[litert] skipped (use --use-litert to enable post-processing)")
            final_text = transcript

        # 5) Output + clipboard
        print("\n================= WYNIK =================\n")
        print(final_text)
        print("\n=========================================\n")

        copied = copy_to_clipboard(final_text, verbose=verbose)
        if not copied:
            print("[warn] Nie udało się skopiować do schowka.", file=sys.stderr)
        else:
            vprint(verbose, "[ok] Wynik skopiowany do schowka.")

        vprint(verbose, f"[info] Zapisany plik audio: {DEFAULT_OUTPUT_WAV}")

        if not ask_continue_recording():
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
