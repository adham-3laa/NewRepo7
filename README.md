# Voice-Controlled Smart Home Demo

English-only demo: speak (or type) English commands, a simulated house updates.

After the first run it works fully offline (except the optional Groq fallback).

## How it works (simple version)

1. Browser records your voice with `st.audio_input` (no PyAudio needed).
2. Local Whisper model `tiny.en` (~75 MB) turns audio into English text.
   It downloads once on the first run, then stays cached and works offline.
3. A tiny scikit-learn classifier (trained in code, no external data)
   turns text into an intent plus confidence. Below 0.60 confidence,
   plain keyword rules decide instead, so weak guesses never trigger actions.
4. The house state updates and the page redraws.

## Run steps (Windows, Python 3.10+, normal laptop mic, CPU only)

1. Install packages (if one pinned version fails, retry that line without `==`):

   ```
   pip install -r requirements.txt
   ```

2. First run needs internet once (model download ~75 MB):

   ```
   streamlit run app.py
   ```

3. Later runs work offline. Just run the same command again.

4. Optional Groq fallback (online only, model `whisper-large-v3-turbo`):
   - Get a free key at https://console.groq.com
   - Set it (never hard-code it): `setx GROQ_API_KEY your_key_here`
   - Restart the terminal, then choose Groq in the sidebar.
   - Free-tier file limit is 25 MB. Our 3-5 second clips are far below that.

## Test order

Typed commands first (no mic needed), then voice (see app sidebar help).
Try: "turn on the kitchen light", "open the door", "lock the door",
"turn off all lights", "what is the status of the house", "make coffee"
(unknown command test). The UI shows the transcript, the intent, and the
confidence so the 0.60 threshold can be tuned in `command_parser.py`.
