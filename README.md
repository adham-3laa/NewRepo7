# Voice-Controlled Smart Home Demo

English-only demo: speak (or type) English commands, a simulated house updates.

After the first run it works fully offline (except the optional Groq fallback).

## How it works (simple version)

1. Press START once in the voice panel: the browser streams the
   microphone continuously to the server over WebRTC
   (`streamlit-webrtc`), no PyAudio needed.
2. The stream is converted to 16 kHz mono and cut into exactly 30 ms
   frames; a voice activity detector (`webrtcvad-wheels`) finds
   utterance boundaries. A command ends after ~0.8 s of pause, and a
   200 ms pre-roll keeps the first phoneme alive. Speech longer than
   15 s is rejected by default (all timings are tunable in the sidebar).
3. Each finished utterance is transcribed in a background thread by the
   local Whisper model `tiny.en` (~75 MB). It downloads once on the
   first run, then stays cached and works offline. The transcription is
   biased with the house vocabulary (`initial_prompt` plus `hotwords`)
   and tuned for short commands. Set `WHISPER_MODEL=base.en` (~145 MB)
   or `small.en` (~480 MB) before restarting for a larger model.
   Benchmark on short commands: base.en matched tiny.en on clean and
   lightly noised clips but was ~2x slower, so tiny.en stays the default.
4. A speech-mistake normalization pass repairs likely transcription errors
   in known room/device/action words before anything else runs (see
   "Tolerant command understanding" below).
5. A tiny scikit-learn classifier (trained in code, no external data)
    turns text into an intent plus confidence. Below 0.65 confidence,
    plain keyword rules decide instead, so weak guesses never trigger actions.
6. The house state updates and the page redraws. Every utterance carries
   a unique id and a generation token, so reruns, sessions and the
   "Reset house" button can never run the same speech twice or execute
   stale audio.

## Run steps (Windows, Python 3.10+, normal laptop mic, CPU only)

1. Install packages (if one pinned version fails, retry that line without `==`):

   ```
   pip install -r requirements.txt
   ```

2. First run needs internet once (model download ~75 MB):

   ```
   streamlit run app.py
   ```

3. In the page, press START in the voice panel and allow the microphone
   when the browser asks. Speak naturally; each command runs on its own
   after a short pause. Press STOP to stop listening (no audio is
   captured while stopped).

4. Later runs work offline. Just run the same command again.

5. Optional Groq fallback (online only, model `whisper-large-v3-turbo`):
   - Get a free key at https://console.groq.com
   - Set it (never hard-code it): `setx GROQ_API_KEY your_key_here`
   - Restart the terminal, then choose Groq in the sidebar.
   - Free-tier file limit is 25 MB. Our short utterances are far below that.

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
- Sidebar keeps engine choice, threshold display, the listener tuning
  panel (pause length, minimum speech, pre-roll, maximum duration, VAD
  mode), accuracy checks (groups a-e), setup help and the session-only
  voice log (Correct/Wrong buttons, nothing written to disk).

## Always-on voice listening (new)

- START the WebRTC stream once and speak naturally; no record button.
  The live status line shows: stopped / starting / listening / speech
  detected / processing, plus the last transcript, engine, how many
  utterances ran, and why audio was discarded.
- Only the Streamlit app thread touches the house: background threads
  segment audio and run Whisper, never commands. A drainer hands each
  utterance result to the existing pipeline exactly once per session.
- "Reset house" keeps listening but bumps a generation token so any
  utterance captured before the reset can never execute afterwards.
- Safety: speech longer than the maximum is rejected and further speech
  is ignored until the next pause; a queue overflow drops the
  compromised utterance until a pause. Errors (mic blocked, stream
  stalled or ended) are shown with the exact next step.
- The page must stay open while listening. Microphone access requires
  `https://` or `localhost`. A public STUN server is preconfigured;
  add a TURN server for strict corporate NATs. `av` is pinned to 17.x
  because `streamlit-webrtc` does not support 18 yet.
- Some Chromium-based browsers never answer the server's RFC 7675
  consent checks, so unpatched aioice drops perfectly healthy sessions
  about 30 s after connecting (aiortc issues #1190/#1235, aioice #58).
  The app disables consent expiry at startup; liveness is covered by
  streamlit-webrtc track-ended events plus the in-app stall warning.
  Set `VOICE_KEEP_CONSENT=1` to keep the library default, or
  `VOICE_DEBUG=1` for verbose WebRTC/aioice server logs.
- `st.audio_input` was removed; the typed box stays as a fallback and
  "Run command" still works as before.

## Test order

Typed commands first (no mic needed), then voice: press START in the
voice panel and allow the microphone, then speak (see app sidebar help).
Try: "turn on the kitchen light", "open the door", "lock the door",
"turn off all lights", "what is the status of the house", "make coffee"
(unknown command test). The UI shows the transcript, the intent, and the
confidence so the 0.65 threshold can be tuned in `command_parser.py`.

## Tolerant command understanding (new)

Speech-recognition mistakes in known vocabulary are repaired before
intent classification and before multi-command splitting, for voice
and typed input alike. The result panel shows `Heard:` (original) and
`Interpreted:` (after repairs) plus the applied corrections.

- Room variants: `bathrom`, `bath room`, `bathrum`, `bedrom`,
  `bed room`, `badroom` (one edit from bedroom), `kichen`, `kitchin` ...
- Device/object variants: `ovem`, `windo`, `windoe`, `lite`,
  `water heather`, `watter heater`, `air condition` ...
- Action variants: `trun on`, `log the door`, `turn on the kitchen like`
- Missing/extra short words are tolerated (`turn on in kitchen light`).
- Repairs are conservative: explicit aliases first, then difflib fuzzy
  matching only when a single vocabulary word is close enough. Protected
  ordinary words (`right`, `store`, `closet`, `statue`, ...) and
  unrelated text (`make coffee`) are never rewritten. A typo that still
  matches two rooms/devices stays refused as ambiguous.

## Multi-command sentences (new)

One spoken utterance or typed line can hold up to 4 commands, run in
spoken order. Each is judged separately; a refused piece never stops
the rest. Keep speech under 15 seconds (longer speech is rejected by
the listener by default) and speak steadily (no long pauses).

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
