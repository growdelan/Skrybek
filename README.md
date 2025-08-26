# Skrybek — lokalne nagrywanie → transkrypcja (mlx‑whisper) → *(opcjonalnie)* obróbka tekstu (Gemma‑3 MLX)

**Skrybek** to prosty, jednoplikowy workflow w Pythonie (`main.py`) dla macOS (Apple Silicon):
1) nagrywa dźwięk z mikrofonu,
2) automatycznie kończy po `3 s` ciszy,
3) zapisuje `recording.wav` (mono, 16 kHz, PCM16),
4) transkrybuje lokalnie przez **mlx‑whisper**,
5) *(opcjonalnie, gdy podasz `--use-gemma`)* obrabia wynik modelem **Gemma‑3** w formacie **MLX** przez `mlx_vlm`,
6) wypisuje i kopiuje wynik do schowka.

Działa całkowicie offline (po jednorazowym pobraniu modeli).

---

## Najważniejsze funkcje
- ✅ Zatrzymanie nagrywania po **ciągłej ciszy** (domyślnie `3.0 s`; konfigurowalne `--silence-threshold` i `--silence-seconds`).
- ✅ Zapis audio **WAV mono 16 kHz, PCM16** do `./recording.wav`.
- ✅ **Transkrypcja lokalna** przez `mlx_whisper.transcribe(...)` — wybór modelu przez `--whisper-model` (używasz domyślnie `mlx-community/whisper-large-v3-turbo`).
- ✅ **Obróbka tekstu** przez `mlx_vlm` jest **opcjonalna** — włączysz ją flagą `--use-gemma`. Bez tej flagi dostajesz czystą transkrypcję.
- ✅ **Schowek**: `pyperclip` z fallbackiem na `pbcopy`.
- ✅ Flagi CLI: `--use-gemma`, `--prompt`, `--silence-threshold`, `--silence-seconds`, `--rate`, `--device`, `--whisper-model`, `--vlm-model`, `--max-tokens`, `--temperature`, `--verbose`.

> `--prompt` ma sens wyłącznie z `--use-gemma` (prompt jest używany w etapie obróbki modelem).

---

## Wymagania
- **System**: macOS na Apple Silicon (MLX).
- **Python**: 3.10+
- **Pakiety**: `sounddevice`, `numpy`, `scipy`, `pyperclip`, `mlx-whisper`, `mlx-vlm`
- **Narzędzia**: `ffmpeg` (dla `mlx-whisper`) – instalacja przez Homebrew:
  ```bash
  brew install ffmpeg
  ```
- **Uprawnienia**: nadaj Terminalowi dostęp do mikrofonu (Ustawienia systemowe → Prywatność i bezpieczeństwo → **Mikrofon**).
- **Audio backend**: `sounddevice`/PortAudio. Listę urządzeń sprawdzisz:
  ```bash
  python -c "import sounddevice as sd; print(sd.query_devices())"
  ```

---

> Modele MLX są pobierane do cache **Hugging Face**: `~/.cache/huggingface/hub`. Lokalizację możesz zmienić przez `HF_HOME` lub `HF_HUB_CACHE`.

---

## Szybki start
```bash
# 1) Tylko transkrypcja (bez Gemmy):
uv run main.py --verbose

# 2) Transkrypcja + obróbka Gemmą (użyj promptu):
uv run main.py --use-gemma 
```

---

## Użycie (wybrane flagi)
- `--use-gemma` – włącz obróbkę przez Gemma‑3 (inaczej zwróci samą transkrypcję).
- `--prompt "..."` – prompt do etapu obróbki (działa tylko z `--use-gemma`).  
- `--silence-threshold 0.01` – próg RMS (0..1) uznawany za ciszę.  
- `--silence-seconds 3.0` – czas ciągłej ciszy kończącej nagrywanie.  
- `--rate 16000` – częstotliwość próbkowania.  
- `--device "MacBook Pro Microphone"` – wybierz mikrofon (patrz `sd.query_devices()`).  
- `--whisper-model mlx-community/whisper-large-v3-turbo` – model dla `mlx-whisper` (Twoja domyślna wartość).  
- `--vlm-model mlx-community/gemma-3-4b-it-qat-4bit` – model dla `mlx_vlm` (używany tylko z `--use-gemma`).  
- `--max-tokens 512`, `--temperature 0.7`, `--verbose` – parametry generacji/logowania.

---

## Modele i rekomendacje

### Transkrypcja (Whisper, MLX)
- **Domyślnie**: `mlx-community/whisper-large-v3-turbo` – dobry kompromis jakość/szybkość.  
- **Maks. jakość**: `mlx-community/whisper-large-v3-mlx`.  
- **Mniej RAM/VRAM**: warianty Q4, np. `mlx-community/whisper-large-v3-turbo-q4`.

### Obróbka tekstu (Gemma‑3 przez `mlx_vlm`)
- Domyślny checkpoint: `mlx-community/gemma-3-4b-it-qat-4bit`.  
- Wejście **tekst‑only**: `apply_chat_template(..., num_images=0)` i `generate(..., image=None)`.

---

## Gdzie trafiają modele i jak je czyścić?
- Cache HF: `~/.cache/huggingface/hub` (domyślnie). Zmień lokalizację przez `HF_HOME` lub `HF_HUB_CACHE`.
- Czyszczenie cache:
  ```bash
  pip install -U "huggingface_hub[cli]"
  hf cache delete          # TUI do wyboru repo/rewizji
  # lub bez TUI:
  hf cache delete --disable-tui --refs mlx-community/whisper-large-v3-turbo
  ```

---

## Rozwiązywanie problemów
- **Brak dźwięku / zły mikrofon** → użyj `--device` lub sprawdź `sd.query_devices()`.  
- **Brak uprawnień do mikrofonu** → dodaj Terminal do sekcji **Mikrofon**.  
- **`ffmpeg` nie znaleziony** → `brew install ffmpeg`.  
- **Schowek nie kopiuje** → sprawdź `pyperclip`; fallback używa `pbcopy`.

---

## Test akceptacyjny (manualny)
1) `python main.py` – wypowiedz 1–2 zdania, zamilknij ≥ 3 s, sprawdź `./recording.wav` i wynik (transkrypcja).  
2) `python main.py --use-gemma --prompt "Zrób zwięzłe streszczenie"` – powtórz test i sprawdź, że wynik pochodzi z obróbki Gemmą i jest w schowku.

---

## Licencja
Kod przykładowy/edukacyjny – dostosuj do swoich potrzeb. Zależności pod własnymi licencjami.
