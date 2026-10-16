# Voice-Controlled Smart Home Demo

English-only demo: speak (or type) English commands, a simulated house updates.

After the first run it works fully offline (except the optional Groq fallback).

## How it works (simple version)

1. Browser records your voice with `st.audio_input` (no PyAudio needed).
2. Local Whisper model `tiny.en` (~75 MB) turns audio into English text.
   It downloads once on the first run, then stays cached and works offline.
3. A tiny scikit-learn classifier (trained in code, no external data)
    turns text into an intent plus confidence. Below 0.65 confidence,
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

## User interface

- Top-down 2D floor plan (the main element): kitchen, bathroom and
  bedroom side by side, front door in the kitchen's north wall, window
  in the bedroom's east wall. Lit rooms glow warm yellow; the oven
  shows heat waves, the water heater a pilot dot, the air conditioner
  airflow dashes; the door leaf swings open/closed with a padlock
  indicator; window panes slide apart when open.
- The drawing is rebuilt from the SmartHome state after every command
  (`floorplan.py` returns an SVG string shown with `st.image`), so it
  can never disagree with the actual state. Rejected commands leave
  the picture unchanged.
- Event monitor below the plan: newest first, max 30 entries, each
  with time, transcript, piece, intent, target, confidence and one of
  SUCCESS / ALREADY IN STATE / REJECTED / UNKNOWN COMMAND.
- Sidebar keeps engine choice, threshold display, accuracy checks
  (groups a-d), setup help and the session-only voice log
  (Correct/Wrong buttons, nothing written to disk).

## Test order

Typed commands first (no mic needed), then voice (see app sidebar help).
Try: "turn on the kitchen light", "open the door", "lock the door",
"turn off all lights", "what is the status of the house", "make coffee"
(unknown command test). The UI shows the transcript, the intent, and the
confidence so the 0.65 threshold can be tuned in `command_parser.py`.

## Multi-command sentences (new)

One recording or typed line can hold up to 4 commands, run in spoken
order. Each is judged separately; a refused piece never stops the rest.
Keep recordings under 15 seconds and speak steadily (no long pauses).

- "turn on the kitchen light and open the window"
- "turn off the oven, then lock the door"
- "turn on the kitchen and bedroom lights" (shared words are filled in)
- "switch on the air conditioner and the bedroom light, then tell me the status"
- Conflicts and ambiguity are refused with a reason, e.g.
  "open the door and lock the door" (second refused: door is open),
  "turn on the kitchen light and the bedroom" (ambiguous piece rejected).
- 5 commands in one sentence: only the first 4 run. Example:
  "turn on the oven, turn on the air conditioner, turn on the kitchen
  light, turn on the water heater, and open the window".
