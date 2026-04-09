# Skrybek — lokalne nagrywanie -> transkrypcja (Whisper lub Parakeet) -> (opcjonalnie) korekta tekstu przez LiteRT (Gemma4 E2B)

**Skrybek** to prosty, jednoplikowy workflow w Pythonie (`main.py`) dla macOS (Apple Silicon):
1) nagrywa dźwięk z mikrofonu,
2) automatycznie kończy po `3 s` ciszy lub naciśnięciu **ESC**,
3) zapisuje `recording.wav` (mono, 16 kHz, PCM16),
4) transkrybuje lokalnie przez **mlx-whisper** albo **mlx-audio (Parakeet)**,
5) opcjonalnie poprawia interpunkcję/wielkie litery przez **LiteRT-LM** (np. `gemma-4-E2B-it`),
6) wypisuje i kopiuje wynik do schowka,
7) pyta w pętli `Czy nagrać kolejną wiadomość [T/N]:` (kolor cyjan).

Działa lokalnie (po jednorazowym pobraniu modeli).

## Najważniejsze funkcje
- Zatrzymanie nagrywania po ciągłej ciszy (`--silence-threshold`, `--silence-seconds`).
- Zapis audio WAV mono 16 kHz, PCM16 do `./recording.wav`.
- Dwa backendy STT: `whisper` przez `mlx-whisper` i `parakeet` przez `mlx-audio`.
- Transkrypcja Whisper z wyborem modelu `--whisper-model`.
- Transkrypcja Parakeet z wyborem modelu `--parakeet-model`.
- Opcjonalna korekta tekstu przez LiteRT po włączeniu `--use-litert`.
- Model LiteRT podawany jawnie przez `--litert-model-path`.
- Schowek: `pyperclip` z fallbackiem `pbcopy`.
- Po każdym przebiegu pytanie `T/N` o nagranie kolejnej wiadomości.

## Wymagania
- **System**: macOS na Apple Silicon.
- **Python**: 3.13+
- **Pakiety**: `sounddevice`, `numpy`, `scipy`, `pyperclip`, `mlx-whisper`, `mlx-audio`, `litert-lm`
- **Narzędzia**: `ffmpeg` (dla `mlx-whisper`):
  ```bash
  brew install ffmpeg
  ```
- **Uprawnienia**: terminal musi mieć dostęp do mikrofonu.

## Backend STT
- `--stt-backend whisper` - obecny backend oparty o `mlx-whisper`.
- `--stt-backend parakeet` - nowy backend oparty o `mlx-audio`.
- `--whisper-model` działa tylko dla backendu `whisper`.
- `--parakeet-model` działa tylko dla backendu `parakeet`.

Domyślne modele:
- Whisper: `mlx-community/whisper-large-v3-turbo`
- Parakeet: `mlx-community/parakeet-tdt-0.6b-v3`

## Model LiteRT (Gemma4 E2B)
Pobierz lokalny plik `.litertlm`, np. z:
- [litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/tree/main)

Przykład użycia modelu lokalnego:
```bash
uv run main.py --use-litert --litert-model-path /pelna/sciezka/do/modelu.litertlm
```

## Szybki start
```bash
# 1) Tylko transkrypcja Whisper (bez LiteRT)
uv run main.py --verbose

# 2) Tylko transkrypcja Parakeet (bez LiteRT)
uv run main.py --stt-backend parakeet --verbose

# 3) Transkrypcja Whisper + korekta przez LiteRT
uv run main.py --use-litert --litert-model-path /pelna/sciezka/do/modelu.litertlm

# 4) Transkrypcja Parakeet + korekta przez LiteRT
uv run main.py --stt-backend parakeet --use-litert --litert-model-path /pelna/sciezka/do/modelu.litertlm

# 5) Jak wyżej, ale z własną instrukcją dla korekty
uv run main.py --use-litert --litert-model-path /pelna/sciezka/do/modelu.litertlm --prompt "Popraw interpunkcję i wielkie litery, bez zmiany sensu."
```

## Użycie (wybrane flagi)
- `--stt-backend whisper|parakeet` - wybór backendu STT.
- `--use-litert` - włącza korektę tekstu przez LiteRT.
- `--litert-model-path /.../model.litertlm` - ścieżka do lokalnego modelu LiteRT (wymagana z `--use-litert`).
- `--prompt "..."` - dodatkowa instrukcja dla etapu LiteRT.
- `--whisper-model ...` - model Whisper używany tylko przez backend `whisper`.
- `--parakeet-model ...` - model Parakeet używany tylko przez backend `parakeet`.
- `--silence-threshold 0.01` - próg RMS ciszy.
- `--silence-seconds 3.0` - czas ciszy kończącej nagranie.
- `--rate 16000` - częstotliwość próbkowania.
- `--device "MacBook Pro Microphone"` - wybór mikrofonu.
- `--whisper-model mlx-community/whisper-large-v3-turbo` - model transkrypcji Whisper.
- `--verbose` - szczegółowe logi.

## Rozwiązywanie problemów
- Brak `--litert-model-path` przy `--use-litert` -> aplikacja kończy działanie z kontrolowanym błędem.
- Brak `mlx-audio` przy `--stt-backend parakeet` -> doinstaluj zależności projektu i uruchom ponownie.
- Brak uprawnień mikrofonu -> włącz dostęp dla terminala w ustawieniach systemu.
- `ffmpeg` nie znaleziony -> `brew install ffmpeg`.
- Schowek nie kopiuje -> fallback używa `pbcopy`.

## Test akceptacyjny (manualny)
1. `uv run main.py` -> sprawdź nagranie, transkrypcję i kopiowanie.
2. `uv run main.py --stt-backend parakeet` -> sprawdź transkrypcję backendem Parakeet.
3. Po pierwszym wyniku wpisz `T` -> powinno zacząć się kolejne nagrywanie bez ponownego wyboru backendu.
4. Wpisz `N` -> aplikacja powinna zakończyć się kodem 0.
5. `uv run main.py --use-litert --litert-model-path /.../model.litertlm` -> sprawdź, że wynik jest po korekcie LiteRT.
6. `uv run main.py --stt-backend parakeet --use-litert --litert-model-path /.../model.litertlm` -> sprawdź ścieżkę Parakeet + LiteRT.
7. `uv run main.py --use-litert` -> sprawdź kontrolowany błąd o braku `--litert-model-path`.

## Licencja
Kod przykładowy/edukacyjny - dostosuj do swoich potrzeb. Zależności pod własnymi licencjami.
