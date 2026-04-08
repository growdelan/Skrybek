# main.py
# macOS (Apple Silicon) | Python 3.13+
# Flow: record mic -> stop after N sec silence OR ESC -> save WAV -> transcribe (mlx-whisper)
# -> (optional) post-process text with LiteRT (gemma4 E2B) -> print + copy to clipboard.

from __future__ import annotations

import argparse
import queue
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

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

DEFAULT_USER_PROMPT = (
    "Przepisz tę wypowiedź dokładnie po polsku. "
    "Zwróć tylko finalny tekst użytkownika. "
    "Nie zmieniaj sensu, nie parafrazuj i niczego nie dopisuj. "
    "Popraw wyłącznie interpunkcję, wielkie litery i oczywistą czytelność zapisu."
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


class LiteRTError(Exception):
    """Bazowy błąd obróbki LiteRT."""


class LiteRTUnavailableError(LiteRTError):
    """Model LiteRT jest niedostępny."""


class LiteRTProcessingError(LiteRTError):
    """LiteRT nie zwrócił poprawnego wyniku."""


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

    def process_text(self, transcript: str, user_prompt_text: Optional[str]) -> str:
        if not transcript.strip():
            return ""

        if self._engine is None:
            raise LiteRTUnavailableError("Model LiteRT nie został poprawnie zainicjalizowany.")

        tool_result: dict[str, str] = {}

        def return_final_text(final_text: str) -> str:
            tool_result["text"] = final_text
            return "OK"

        task_prompt = (user_prompt_text or DEFAULT_USER_PROMPT).strip() or DEFAULT_USER_PROMPT
        content = [
            {
                "type": "text",
                "text": (
                    f"{task_prompt}\n\n"
                    "TRANSKRYPCJA (nie zmieniaj sensu wypowiedzi):\n"
                    f"{transcript}"
                ),
            }
        ]

        try:
            with self._engine.create_conversation(
                messages=[{"role": "system", "content": SYSTEM_PROMPT}],
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
def transcribe_with_mlx_whisper(
    wav_path: str,
    whisper_model: Optional[str] = None,
    verbose: bool = False,
) -> str:
    """
    Używa oficjalnego API: mlx_whisper.transcribe(file, path_or_hf_repo=...) -> dict['text'].
    """
    import mlx_whisper  # PyPI: mlx-whisper

    kwargs = {}
    if whisper_model:
        kwargs["path_or_hf_repo"] = whisper_model
    kwargs["language"] = "pl"

    vprint(
        verbose,
        f"[whisper] transcribing {wav_path} with model={whisper_model or 'default'} …",
    )
    out = mlx_whisper.transcribe(wav_path, **kwargs)
    text = (out.get("text") or "").strip()
    segments = out.get("segments") or []
    if not text:
        vprint(verbose, "[whisper] empty transcript")
    else:
        seg_info = f", segments={len(segments)}" if segments else ""
        vprint(verbose, f"[whisper] transcript len={len(text)} chars{seg_info}")
    return text


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
        description="Record -> Transcribe (mlx-whisper) -> (optional) Process (LiteRT) -> Print + Clipboard"
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
        help="User prompt for LiteRT post-processing (used only with --use-litert).",
    )
    ap.add_argument(
        "--whisper-model",
        type=str,
        default="mlx-community/whisper-large-v3-turbo",
        help="Whisper model id/path for mlx-whisper (e.g., mlx-community/whisper-large-v3-turbo)",
    )
    ap.add_argument("--verbose", action="store_true", help="Verbose logs")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    verbose = args.verbose

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

        # 3) Transcribe locally (mlx-whisper)
        transcript = transcribe_with_mlx_whisper(
            DEFAULT_OUTPUT_WAV, whisper_model=args.whisper_model, verbose=verbose
        )

        # 4) (Optional) Process text with LiteRT
        if args.use_litert:
            if not args.litert_model_path:
                print(
                    "[ERR] Brak --litert-model-path. Podaj ścieżkę do lokalnego pliku .litertlm.",
                    file=sys.stderr,
                )
                return 2

            processor: LiteRTTextProcessor | None = None
            try:
                processor = LiteRTTextProcessor(args.litert_model_path, verbose=verbose)
                final_text = processor.process_text(
                    transcript=transcript,
                    user_prompt_text=args.prompt,
                )
            except LiteRTError as exc:
                print(f"[litert][ERR] {exc}", file=sys.stderr)
                return 1
            finally:
                if processor is not None:
                    processor.close()
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
