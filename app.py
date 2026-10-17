"""Streamlit web interface for the voice-controlled smart home."""

import logging
import os
import time

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from command_parser import CONFIDENCE_THRESHOLD, evaluate_with_rules
from floorplan import write_floorplan
from home import SmartHome
from intent_model import evaluate_accuracy, evaluate_single_v2, load_or_train
from listener import (
    ListenerHandle,
    ListenerWorker,
    ResultDrainer,
    SegmenterConfig,
)
from multi_command import (
    MAX_COMMANDS,
    execute_multi,
    parse_multi,
    score_multi_test,
)
from speech import (
    groq_key_available,
    load_local_model,
    local_model_name,
    local_model_size_text,
    transcribe_groq,
    transcribe_local,
)
from training_data import (
    get_garbage_test,
    get_single_test_v2,
    get_stt_variant_test,
)


@st.cache_resource
def get_intent_model():
    """Train or load the tiny intent classifier once (takes seconds)."""
    return load_or_train()


@st.cache_resource
def get_whisper_model():
    """Load the local Whisper model once. First run downloads it."""
    name = local_model_name()
    with st.spinner("Loading local voice model %s (first run downloads %s, "
                     "then works offline)..." % (name,
                                                 local_model_size_text(name))):
        return load_local_model()


def _setup_debug_logging():
    """Verbose WebRTC logs on the server console when VOICE_DEBUG=1.

    Off by default; useful when diagnosing microphone or connection
    problems (the library logs track start/end and ICE state changes).
    """
    if os.environ.get("VOICE_DEBUG") != "1":
        return
    handler = logging.StreamHandler()  # stderr
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s"))
    for name in ("streamlit_webrtc", "aioice.ice", "aioice.stun"):
        logger = logging.getLogger(name)
        if logger.handlers:
            continue
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)


def _patch_ice_consent():
    """Stop aioice from tearing down sessions whose browser ignores it.

    RFC 7675 consent checks are sent every ~5 s and aioice closes the ICE
    connection after 6 unanswered checks (~30 s).  Several Chromium-based
    browsers never answer aioice's consent requests while media keeps
    flowing, so healthy sessions die ~30 s in (aiortc issues #1190/#1235,
    aioice issue #58).  Media liveness here is already handled by
    streamlit-webrtc track-ended events and the in-app stall warning, so
    consent expiry is disabled by default for a reliable always-on
    listener.  Set VOICE_KEEP_CONSENT=1 to restore the library default.
    """
    if os.environ.get("VOICE_KEEP_CONSENT") == "1":
        return
    from aioice.ice import Connection

    if getattr(Connection.query_consent, "_voice_disabled", False):
        return

    async def _query_consent_disabled(self):
        return

    _query_consent_disabled._voice_disabled = True
    Connection.query_consent = _query_consent_disabled


_patch_ice_consent()


def init_state():
    """Create house, history and model state once per session."""
    if st.session_state.get("v") != 2:
        # Version 2 changed the result/history format: reset old sessions
        # so unpacking never crashes after an app update.
        st.session_state.home = SmartHome()
        st.session_state.history = []  # List of (transcript, summary).
        st.session_state.last_result = None
        st.session_state.v = 2
    if "home" not in st.session_state:
        st.session_state.home = SmartHome()
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "events" not in st.session_state:
        # Chronological individual action outcomes, newest last.
        # Added without touching home/history/voice_log.
        st.session_state.events = []
    # Always-on listener: the handle must live for the whole session
    # because the WebRTC audio callback captures it permanently.
    if "listener_handle" not in st.session_state:
        st.session_state.listener_handle = ListenerHandle()
    if "listener_drainer" not in st.session_state:
        st.session_state.listener_drainer = ResultDrainer()
    if "listener_worker" not in st.session_state:
        st.session_state.listener_worker = None
    if "listener_settings" not in st.session_state:
        st.session_state.listener_settings = None
    if "listener_playing" not in st.session_state:
        st.session_state.listener_playing = False
    if "listener_last_transcript" not in st.session_state:
        st.session_state.listener_last_transcript = ""
    if "listener_last_error" not in st.session_state:
        st.session_state.listener_last_error = ""


def intent_target(intent):
    """Presentation-only (room/device, action) pair for an intent."""
    table = {
        "kitchen_light_on": ("Kitchen", "light on"),
        "kitchen_light_off": ("Kitchen", "light off"),
        "bathroom_light_on": ("Bathroom", "light on"),
        "bathroom_light_off": ("Bathroom", "light off"),
        "bedroom_light_on": ("Bedroom", "light on"),
        "bedroom_light_off": ("Bedroom", "light off"),
        "kitchen_device_on": ("Kitchen oven", "turn on"),
        "kitchen_device_off": ("Kitchen oven", "turn off"),
        "bathroom_device_on": ("Water heater", "turn on"),
        "bathroom_device_off": ("Water heater", "turn off"),
        "bedroom_device_on": ("Air conditioner", "turn on"),
        "bedroom_device_off": ("Air conditioner", "turn off"),
        "door_open": ("Front door", "open"),
        "door_close": ("Front door", "close"),
        "door_lock": ("Front door", "lock"),
        "door_unlock": ("Front door", "unlock"),
        "window_open": ("Window", "open"),
        "window_close": ("Window", "close"),
        "all_lights_off": ("All lights", "turn off"),
        "everything_off": ("Everything", "turn off"),
        "status": ("House", "status"),
    }
    return table.get(intent, ("-", "-"))


def handle_text(text, model, kind="typed"):
    """Parse (possibly multi-command) text, run it, store results."""
    if not (text or "").strip():
        st.session_state.last_result = {
            "transcript": "", "corrected": "", "corrections": [],
            "limited": False, "kind": kind,
            "voted": False, "outcomes": [{
                "piece": "", "intent": "unknown", "confidence": 0.0,
                "source": "unknown", "corrections": [],
                "message": "I heard nothing. Please speak closer "
                           "or type a command.", "tag": "not understood"}]}
        return
    parsed = parse_multi(text, model)
    outcomes = execute_multi(st.session_state.home, parsed)
    st.session_state.last_result = {
        "transcript": parsed["transcript"],
        "corrected": parsed.get("corrected", parsed["transcript"]),
        "corrections": parsed.get("corrections", []),
        "limited": parsed["limited"],
        "kind": kind, "voted": False, "outcomes": outcomes}
    counts = {}
    for outcome in outcomes:
        counts[outcome["tag"]] = counts.get(outcome["tag"], 0) + 1
    summary = "%d command(s): %s" % (
        len([o for o in outcomes if o["piece"] != "..."]),
        ", ".join("%d %s" % (n, tag) for tag, n in sorted(counts.items())))
    st.session_state.history.append((parsed["transcript"], summary))
    st.session_state.history = st.session_state.history[-10:]
    append_events(parsed["transcript"], outcomes)


def append_events(transcript, outcomes):
    """Record one event per action outcome, exactly once per command.

    Called only from handle_text (the execution point), never from
    rendering, so reruns cannot duplicate events. Keeps the latest 30.
    """
    import datetime
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    status_of = {"success": "SUCCESS", "already done": "ALREADY IN STATE",
                 "refused": "REJECTED", "not understood": "UNKNOWN COMMAND"}
    for outcome in outcomes:
        target_room, target_action = intent_target(outcome["intent"])
        st.session_state.events.append({
            "time": stamp, "transcript": transcript,
            "piece": outcome["piece"], "intent": outcome["intent"],
            "target": "%s - %s" % (target_room, target_action),
            "confidence": outcome["confidence"],
            "source": outcome["source"], "message": outcome["message"],
            "status": status_of.get(outcome["tag"], outcome["tag"])})
    st.session_state.events = st.session_state.events[-30:]


_TAG_ICONS = {"success": "\u2705", "already done": "\u26aa",
              "refused": "\u26d4", "not understood": "\u2753"}


def render_results():
    """Show the transcript plus a numbered list of per-command results."""
    result = st.session_state.last_result
    if result is None:
        return
    st.subheader("Result")
    st.write("Heard: '%s'" % result["transcript"])
    if result.get("corrected") and result["corrected"] != result["transcript"]:
        st.write("Interpreted: '%s'" % result["corrected"])
    if result.get("corrections"):
        fixes = ", ".join("%s->%s" % (old, new)
                          for old, new in result["corrections"])
        st.write("Speech corrections: %s" % fixes)
    for number, outcome in enumerate(result["outcomes"], start=1):
        icon = _TAG_ICONS.get(outcome["tag"], "")
        st.write("%d. '%s' -> %s (confidence %.2f, via %s): %s %s" % (
            number, outcome["piece"], outcome["intent"],
            outcome["confidence"], outcome["source"],
            icon, outcome["message"]))
        if outcome["corrections"]:
            fixes = ", ".join("%s->%s" % (old, new)
                              for old, new in outcome["corrections"])
            st.write("   Corrections applied: %s" % fixes)
    st.write("Threshold: %.2f (each command judged separately)" %
             CONFIDENCE_THRESHOLD)
    render_verdict_buttons()


def render_verdict_buttons():
    """Let the user mark a voice attempt correct/wrong (session only).

    Nothing is written to disk. The log lives in st.session_state and
    vanishes when the session ends.
    """
    result = st.session_state.last_result
    if result is None or result.get("kind") != "voice" \
            or result.get("voted"):
        return
    st.write("Was this voice attempt correct?")
    col_yes, col_no = st.columns(2)
    if col_yes.button("Correct"):
        _record_verdict(True)
    if col_no.button("Wrong"):
        _record_verdict(False)


def _record_verdict(correct):
    """Store one voice verdict and hide the buttons for this result."""
    if "voice_log" not in st.session_state:
        st.session_state.voice_log = []
    result = st.session_state.last_result
    intents = [o["intent"] for o in result["outcomes"]]
    st.session_state.voice_log.append(
        (result["transcript"], intents, correct))
    result["voted"] = True


def render_voice_tally():
    """Show the session-only voice accuracy tally in the sidebar."""
    log = st.session_state.get("voice_log", [])
    if not log:
        st.sidebar.write("Voice log: no votes yet.")
        return
    good = sum(1 for _, _, correct in log if correct)
    st.sidebar.write("Voice log (this session only): %d/%d correct." %
                     (good, len(log)))
    with st.sidebar.expander("Voice attempts"):
        for transcript, intents, correct in log:
            mark = "OK" if correct else "WRONG"
            st.write("%s '%s' -> %s" % (mark, transcript,
                                        ", ".join(intents)))


def render_accuracy(model):
    """Run test groups (a)-(d) and show the scores in the sidebar."""
    acc_a, _ = evaluate_single_v2(model)
    st.sidebar.write("(a) Singles, classifier alone: %.0f%%" % (acc_a * 100))
    acc_b, _ = evaluate_with_rules(model, get_single_test_v2())
    st.sidebar.write("(b) Singles, end to end: %.0f%%" % (acc_b * 100))
    acc_d, _ = evaluate_with_rules(model, get_garbage_test())
    st.sidebar.write("(d) Garbage safely rejected: %.0f%%" % (acc_d * 100))
    acc_e, _ = evaluate_with_rules(model, get_stt_variant_test())
    st.sidebar.write("(e) STT variants end to end: %.0f%%" % (acc_e * 100))
    cmd_acc, sent_acc, _ = score_multi_test(model)
    st.sidebar.write("(c) Multi per-command: %.0f%%, per-sentence: %.0f%%" %
                     (cmd_acc * 100, sent_acc * 100))
    old_acc, _ = evaluate_accuracy(model)
    st.sidebar.write("Old set (regression): %.0f%%" % (old_acc * 100))


def render_house(home):
    """Draw the top-down floor plan from the SmartHome booleans."""
    on_count = sum(home.lights.values()) + sum(home.devices.values())
    if on_count == 0:
        overall = "ALL OFF"
    else:
        overall = "%d device(s) ON" % on_count
    st.subheader("House overview - %s" % overall)
    st.image(write_floorplan(home), use_container_width=True)


_STATUS_ICONS = {"SUCCESS": "\u2705", "ALREADY IN STATE": "\u26aa",
                 "REJECTED": "\u26d4", "UNKNOWN COMMAND": "\u2753"}


def render_event_feed():
    """Chronological event monitor, newest first. Reads only."""
    st.subheader("Event monitor (last %d)" % len(st.session_state.events))
    if not st.session_state.events:
        st.write("No commands yet. Type or speak a command below.")
        return
    for event in reversed(st.session_state.events):
        icon = _STATUS_ICONS.get(event["status"], "")
        st.write("%s [%s] %s" % (icon, event["status"], event["time"]))
        st.text("Heard: '%s'" % event["transcript"])
        st.text("Piece: '%s' -> %s (%s)" % (
            event["piece"], event["intent"], event["target"]))
        st.text("Confidence %.2f via %s. %s" % (
            event["confidence"], event["source"], event["message"]))


def make_transcribe_fn(engine):
    """Transcription callable for the background listener worker.

    The local Whisper model is loaded here, on the application thread
    (cached across sessions), so the worker thread only runs pure
    inference on a temporary WAV file.
    """
    if engine.startswith("Groq"):
        return lambda wav_path: transcribe_groq(wav_path)
    whisper_model = get_whisper_model()
    return lambda wav_path: transcribe_local(wav_path, whisper_model)


def ensure_listener(handle, playing, engine, config):
    """Keep exactly one worker attached while the stream is playing.

    Called on every rerun from the application thread. Creates the
    worker when the stream starts, rebuilds it when the engine or the
    tuning settings change, and stops it when the stream stops. The
    handle stays the same forever, so the audio callback can never
    route frames to a stale worker.
    """
    worker = st.session_state.listener_worker
    settings = (engine, config.snapshot())
    if not playing:
        if worker is not None and worker.is_running():
            handle.stop()
            worker.stop()
        st.session_state.listener_worker = None
        st.session_state.listener_settings = None
        return None
    if (worker is not None and worker.is_running()
            and st.session_state.listener_settings == settings):
        return worker
    label = "groq" if engine.startswith("Groq") else "local"
    new_worker = ListenerWorker(make_transcribe_fn(engine),
                                engine_label=label, config=config)
    old = handle.swap_worker(new_worker)
    if old is not None:
        old.stop()
    st.session_state.listener_worker = new_worker
    st.session_state.listener_settings = settings
    return new_worker


def listener_status(worker, handle, playing):
    """Compute the (level, text) shown in the live status line."""
    if worker is None or not worker.is_running():
        return ("stopped", "Listener stopped. Press START above, allow "
                           "the microphone, then speak.")
    if not playing:
        return ("starting", "Stream is starting...")
    if handle.conversion_error():
        return ("error", "Audio conversion failed: %s"
                % handle.conversion_error())
    if handle.audio_ended():
        return ("error", "The microphone stream ended unexpectedly. Press "
                         "STOP, then START, and allow the microphone again.")
    status = worker.status()
    now = time.monotonic()
    if status["frames"] == 0:
        if now - status["created_at"] < 10:
            return ("starting", "Starting: allow the microphone in the "
                                "browser when asked.")
        return ("error", "No audio is arriving. Allow the microphone in the "
                         "browser; use https or localhost.")
    stalled_for = now - (status["last_frame_at"] or now)
    if stalled_for > 5:
        return ("error", "Audio stopped arriving %.0f s ago. Press STOP, "
                         "then START again." % stalled_for)
    if status["pending"] > 0:
        return ("processing", "Processing speech...")
    if status["segment_state"] == "suppressed":
        return ("listening", "Listening - waiting for a pause to resync.")
    if status["segment_state"] == "speech":
        return ("speech", "Speech detected - keep talking...")
    return ("listening", "Listening - speak a command.")


_LISTENER_ICONS = {"stopped": "\u26aa", "starting": "\u23f3",
                   "listening": "\U0001F7E2", "speech": "\U0001F3A4",
                   "processing": "\u2699\ufe0f", "error": "\u26d4"}


@st.fragment(run_every="0.5s")
def render_listener_panel(model):
    """Poll the worker, run new voice commands, show the live status.

    Runs every 0.5 s on the application thread. Finished transcriptions
    are handed to the existing handle_text pipeline exactly once (the
    ResultDrainer deduplicates by utterance id), then one full rerun
    refreshes the house view, results and history.
    """
    handle = st.session_state.listener_handle
    drainer = st.session_state.listener_drainer
    worker = st.session_state.listener_worker
    playing = st.session_state.listener_playing

    if worker is not None and worker.is_running() and handle.audio_ended():
        status = worker.status()
        if status["frames"] and time.monotonic() - status["last_frame_at"] < 2:
            # An end event can be left over from a former connection
            # (STOP, page reload, server restart). Frames are provably
            # flowing again, so the flag must not stick.
            handle.clear_audio_ended()

    executed = False
    for result in drainer.drain(worker):
        if result.get("error"):
            st.session_state.listener_last_error = result["error"]
            continue
        st.session_state.listener_last_error = ""
        st.session_state.listener_last_transcript = result["transcript"]
        handle_text(result["transcript"], model, kind="voice")
        executed = True

    level, text = listener_status(worker, handle, playing)
    st.write("%s %s" % (_LISTENER_ICONS.get(level, ""), text))
    if st.session_state.listener_last_error:
        st.error(st.session_state.listener_last_error)
    if st.session_state.listener_last_transcript:
        st.write("Last heard: '%s'"
                 % st.session_state.listener_last_transcript)
    if worker is not None and worker.is_running():
        status = worker.status()
        details = ["engine: %s" % status["engine"],
                   "audio frames: %d" % status["frames"],
                   "utterances transcribed: %d" % status["processed"]]
        if status["discarded"]:
            details.append("discarded: %d" % status["discarded"])
        st.caption(" | ".join(details)
                   + (" - %s" % status["last_note"] if status["last_note"]
                      else ""))

    if executed:
        st.rerun(scope="app")


def main():
    """Build the page: sidebar, house, mic widget, results, history."""
    st.set_page_config(page_title="Voice-Controlled Smart Home", layout="wide")
    st.title("Voice-Controlled Smart Home Demo")
    _setup_debug_logging()
    init_state()
    model = get_intent_model()

    # Sidebar: engine choice and help.
    st.sidebar.header("Settings")
    engine = st.sidebar.radio("Recognition engine",
                              ["Local Whisper (offline)", "Groq (online fallback)"])
    st.sidebar.write("Confidence threshold: %.2f" % CONFIDENCE_THRESHOLD)
    groq_status = "SET" if groq_key_available() else "NOT SET"
    st.sidebar.write("Groq key status: %s (never shown)" % groq_status)
    local_name = local_model_name()
    with st.sidebar.expander("Listener settings"):
        silence_ms = st.slider("Pause that ends a command (ms)",
                               300, 2000, 800, 100)
        min_speech_ms = st.slider("Ignore speech shorter than (ms)",
                                  100, 1000, 250, 50)
        preroll_ms = st.slider("Audio kept before speech starts (ms)",
                               0, 500, 200, 50)
        max_utterance_ms = st.slider("Reject commands longer than (ms)",
                                     3000, 30000, 15000, 500)
        vad_mode = st.slider("VAD aggressiveness (higher rejects "
                             "background noise more)", 0, 3, 2, 1)
        emit_truncated = st.checkbox("Transcribe over-long speech anyway "
                                     "(it may cut off)", value=False)
    config = SegmenterConfig(silence_ms=silence_ms,
                             min_speech_ms=min_speech_ms,
                             preroll_ms=preroll_ms,
                             max_utterance_ms=max_utterance_ms,
                             vad_mode=vad_mode,
                             emit_truncated=emit_truncated)
    with st.sidebar.expander("Setup help"):
        st.write("1. Press START in the voice panel and allow the "
                 "microphone when the browser asks.")
        st.write("2. Speak naturally; each command runs on its own after "
                 "a short pause. No button press needed.")
        st.write("3. Press STOP to stop listening; no audio is captured "
                 "then. Settings changes apply while listening.")
        st.write("4. First run needs internet once to download %s "
                 "(%s)." % (local_name, local_model_size_text(local_name)))
        st.write("5. After that, Local Whisper works fully offline. Model "
                 "can be changed before restarting: "
                 "`setx WHISPER_MODEL base.en` (~145 MB) or `small.en` "
                 "(~480 MB) for more accuracy; tiny.en (current) is "
                 "the fastest.")
        st.write("6. Groq needs GROQ_API_KEY set, e.g. "
                 "`setx GROQ_API_KEY your_key`, then restart.")
        st.write("7. Speak close to the mic, in a quiet room. Up to %d "
                 "commands per sentence: steady speech, max 15 seconds, "
                 "no long pauses." % MAX_COMMANDS)
    if st.sidebar.button("Check accuracy (groups a-e)"):
        render_accuracy(model)
    render_voice_tally()

    # Fixed containers keep the top-level element sequence above the
    # WebRTC component identical on every rerun. Without this, the first
    # command (which adds event-feed entries above the component) shifts
    # the component's position in the tree, the browser remounts the
    # component iframe, and the live audio stream dies mid-session.
    with st.container():
        render_house(st.session_state.home)
    with st.container():
        render_event_feed()

    # Always-on voice control. The WebRTC audio callback only converts
    # and routes frames (never transcribes); the fragment below drains
    # finished utterances and runs them through the existing pipeline.
    st.subheader("Voice control (always on)")
    handle = st.session_state.listener_handle

    def on_audio_frame(frame):
        handle.push_frame(frame)
        return frame

    webrtc_ctx = webrtc_streamer(
        key="voice-listener",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": False, "audio": True},
        media_toggle_controls=False,
        # The processed audio track must stay connected to the peer
        # connection: with sendback_audio=False the library blocks the
        # track and never pulls frames, so the callback never fires.
        # A muted, hidden playback element keeps the sendback silent.
        sendback_audio=True,
        audio_html_attrs={"autoPlay": True, "controls": False,
                          "muted": True, "style": {"display": "none"}},
        rtc_configuration={"iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}]},
        audio_frame_callback=on_audio_frame,
        on_audio_ended=handle.mark_audio_ended,
    )
    playing = bool(webrtc_ctx.state.playing)
    st.session_state.listener_playing = playing
    ensure_listener(handle, playing, engine, config)
    render_listener_panel(model)

    # Typed fallback for machines without a microphone.
    st.subheader("Give a command")
    typed = st.text_input("Or type a command (works without a microphone)",
                          placeholder="turn on the kitchen light")
    col_run, col_reset = st.columns(2)
    run_pressed = col_run.button("Run command")
    if col_reset.button("Reset house"):
        st.session_state.home = SmartHome()
        st.session_state.history = []
        st.session_state.events = []
        st.session_state.last_result = None
        # Keep listening, but invalidate every utterance captured
        # before the reset so it can never execute afterwards.
        if st.session_state.listener_worker is not None:
            st.session_state.listener_worker.reset()
        st.session_state.listener_last_transcript = ""
        st.session_state.listener_last_error = ""
        st.rerun()

    if run_pressed:
        if typed.strip():
            handle_text(typed, model, kind="typed")
        else:
            st.warning("Type a command first.")
        st.rerun()

    # Show what was heard, per-command results, and history.
    render_results()

    if st.session_state.history:
        st.subheader("History (last %d)" % len(st.session_state.history))
        for transcript, summary in reversed(st.session_state.history):
            st.write("- '%s': %s" % (transcript, summary))

    with st.expander("Example commands"):
        st.write("- turn on the kitchen light / switch off the bedroom light")
        st.write("- turn on the oven / turn off the air conditioner")
        st.write("- open the door / close the window / lock the door")
        st.write("- turn off all lights / turn everything off")
        st.write("- what is the status of the house")
        st.write("- Multi: turn on the kitchen light and open the window")
        st.write("- Multi: turn on the kitchen and bedroom lights")
        st.write("- Max %d commands per sentence; keep speech under "
                 "15 seconds." % MAX_COMMANDS)


if __name__ == "__main__":
    main()
