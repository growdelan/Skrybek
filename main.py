# main.py
# macOS (Apple Silicon) | Python 3.10+
# Flow: record mic -> stop after N sec silence OR ESC -> save WAV -> transcribe (mlx-whisper)
# -> (optional) process with Gemma 3 via mlx_vlm -> print + copy to clipboard.

from __future__ import annotations
import argparse
import sys
import time
import queue
import subprocess
from typing import List, Optional

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

# --- Konfiguracja/placeholder ---
USER_PROMPT = f"""
Jesteś Formatorem polskiego tekstu z dyktatu.

Twoje zadania:
- Poprawiaj interpunkcję, wielkie litery i składnię.
- Każde zdanie zaczynaj wielką literą, kończ kropką.
- Jeśli tekst zawiera wyliczenia, instrukcje, kroki, przykłady lub elementy porządkowe:
  - Przerób je na listę punktowaną (- ) lub numerowaną (1., 2., 3.), zależnie od kontekstu.
  - Każdy punkt listy w osobnym wierszu.
- Dziel dłuższy tekst na akapity dla lepszej czytelności.
- Usuwaj powtórzenia wynikające z dyktatu (np. "yyy", "eee", "no no").
- Zachowaj oryginalne słownictwo użytkownika, nie zmieniaj sensu.
- Nie dodawaj komentarzy, podsumowań ani pozdrowień.
- Styl tekstu ma być prosty, klarowny i naturalny.

Cel: przejrzysty, poprawny językowo tekst w formie gotowej do czytania lub publikacji.
"""
DEFAULT_OUTPUT_WAV = "recording.wav"


def vprint(verbose: bool, *args, **kwargs):
    if verbose:
        print(*args, **kwargs, flush=True)


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

    vprint(
        verbose,
        f"[whisper] transcribing {wav_path} with model={whisper_model or 'default'} …",
    )
    out = mlx_whisper.transcribe(wav_path, **kwargs)
    text = (out.get("text") or "").strip()
    if not text:
        vprint(verbose, "[whisper] empty transcript")
    else:
        vprint(verbose, f"[whisper] transcript len={len(text)} chars")
    return text


# --- Obróbka przez Gemma-3 przez mlx_vlm (tekst-only) ---
def process_with_gemma(
    transcript: str,
    user_prompt_text: Optional[str],
    model_id: str = "mlx-community/gemma-3-4b-it-qat-4bit",
    max_tokens: int = 512,
    temperature: float = 0.7,
    verbose: bool = False,
) -> str:
    """
    Ładuje model przez mlx_vlm i uruchamia generację w trybie tekstowym.
    Nowsze wersje mlx_vlm zwracają GenerationResult (.text).
    """
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template

    try:
        from mlx_vlm.utils import load_config
    except Exception:
        load_config = None

    if not transcript.strip():
        return ""

    if verbose:
        print(f"[gemma] loading model: {model_id}", flush=True)
    model, processor = load(model_id)
    config = load_config(model_id) if load_config else getattr(model, "config", None)

    base_prompt = (user_prompt_text or USER_PROMPT or "").strip()
    if base_prompt == "" or base_prompt == "<WSTAW_SWÓJ_PROMPT_TUTAJ>":
        base_prompt = (
            "Przetwórz transkrypcję: streść najważniejsze punkty zwięźle i po polsku."
        )

    prompt = (
        f"{base_prompt}\n\n---\nTRANSKRYPCJA (kontekst użytkownika):\n{transcript}\n"
    )

    formatted = apply_chat_template(processor, config, prompt, num_images=0)

    if verbose:
        print("[gemma] generating …", flush=True)

    out = generate(
        model,
        processor,
        formatted,
        image=None,  # tekst-only
        max_tokens=max_tokens,
        temperature=temperature,
        verbose=verbose,
    )

    if hasattr(out, "text"):
        text = out.text
    elif isinstance(out, dict) and "text" in out:
        text = str(out["text"])
    elif isinstance(out, str):
        text = out
    else:
        text = str(out)
    return text.strip()


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


# --- CLI / main ---
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Record -> Transcribe (mlx-whisper) -> (optional) Process (Gemma-3 via mlx_vlm) -> Print + Clipboard"
    )
    ap.add_argument(
        "--use-gemma",
        action="store_true",
        help="If set, post-process transcript with Gemma via mlx_vlm; otherwise output raw transcript.",
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
        help="User prompt for Gemma post-processing (used only with --use-gemma; overrides USER_PROMPT)",
    )
    ap.add_argument(
        "--whisper-model",
        type=str,
        default="mlx-community/whisper-large-v3-turbo",
        help="Whisper model id/path for mlx-whisper (e.g., mlx-community/whisper-large-v3-turbo)",
    )
    ap.add_argument(
        "--vlm-model",
        type=str,
        default="mlx-community/gemma-3-4b-it-qat-4bit",
        help="mlx_vlm model id (used only with --use-gemma)",
    )
    ap.add_argument(
        "--max-tokens", type=int, default=512, help="Max tokens for VLM generation"
    )
    ap.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature for VLM"
    )
    ap.add_argument("--verbose", action="store_true", help="Verbose logs")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    verbose = args.verbose

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

    # 4) (Optional) Process text with Gemma-3 via mlx_vlm
    if args.use_gemma:
        final_text = process_with_gemma(
            transcript=transcript,
            user_prompt_text=args.prompt,
            model_id=args.vlm_model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            verbose=verbose,
        )
    else:
        vprint(verbose, "[gemma] skipped (use --use-gemma to enable post-processing)")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
