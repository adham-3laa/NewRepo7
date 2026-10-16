"""Streamlit web interface for the voice-controlled smart home."""

import os
import tempfile

import streamlit as st

from command_parser import CONFIDENCE_THRESHOLD, evaluate_with_rules
from home import SmartHome
from intent_model import evaluate_accuracy, evaluate_single_v2, load_or_train
from multi_command import (
    MAX_COMMANDS,
    execute_multi,
    parse_multi,
    score_multi_test,
)
from speech import (
    groq_key_available,
    load_local_model,
    transcribe_groq,
    transcribe_local,
)
from training_data import get_garbage_test, get_single_test_v2


@st.cache_resource
def get_intent_model():
    """Train or load the tiny intent classifier once (takes seconds)."""
    return load_or_train()


@st.cache_resource
def get_whisper_model():
    """Load local tiny.en once. First run downloads ~75 MB."""
    with st.spinner("Loading local voice model tiny.en (first run downloads "
                     "about 75 MB, then works offline)..."):
        return load_local_model()


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


def handle_text(text, model, kind="typed"):
    """Parse (possibly multi-command) text, run it, store results."""
    if not (text or "").strip():
        st.session_state.last_result = {
            "transcript": "", "limited": False, "kind": kind,
            "voted": False, "outcomes": [{
                "piece": "", "intent": "unknown", "confidence": 0.0,
                "source": "unknown", "corrections": [],
                "message": "I heard nothing. Please speak closer "
                           "or type a command.", "tag": "not understood"}]}
        return
    parsed = parse_multi(text, model)
    outcomes = execute_multi(st.session_state.home, parsed)
    st.session_state.last_result = {
        "transcript": parsed["transcript"], "limited": parsed["limited"],
        "kind": kind, "voted": False, "outcomes": outcomes}
    counts = {}
    for outcome in outcomes:
        counts[outcome["tag"]] = counts.get(outcome["tag"], 0) + 1
    summary = "%d command(s): %s" % (
        len([o for o in outcomes if o["piece"] != "..."]),
        ", ".join("%d %s" % (n, tag) for tag, n in sorted(counts.items())))
    st.session_state.history.append((parsed["transcript"], summary))
    st.session_state.history = st.session_state.history[-10:]


_TAG_ICONS = {"success": "\u2705", "already done": "\u26aa",
              "refused": "\u26d4", "not understood": "\u2753"}


def render_results():
    """Show the transcript plus a numbered list of per-command results."""
    result = st.session_state.last_result
    if result is None:
        return
    st.subheader("Result")
    st.write("Heard: '%s'" % result["transcript"])
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
    cmd_acc, sent_acc, _ = score_multi_test(model)
    st.sidebar.write("(c) Multi per-command: %.0f%%, per-sentence: %.0f%%" %
                     (cmd_acc * 100, sent_acc * 100))
    old_acc, _ = evaluate_accuracy(model)
    st.sidebar.write("Old set (regression): %.0f%%" % (old_acc * 100))


def render_house(home):
    """Show door, window and three room cards with icons and colors."""
    door_icon = "🚪 OPEN" if home.door_open else "🚪 CLOSED"
    lock_icon = "🔒 LOCKED" if home.door_locked else "🔓 UNLOCKED"
    window_icon = "🪟 OPEN" if home.window_open else "🪟 CLOSED"
    st.subheader("House overview")
    col_a, col_b = st.columns(2)
    col_a.metric("Front door", "%s (%s)" % (door_icon, lock_icon))
    col_b.metric("Window", window_icon)

    cards = st.columns(3)
    for column, room in zip(cards, ("kitchen", "bathroom", "bedroom")):
        light = "💡 ON" if home.lights[room] else "⚫ OFF"
        device_name = SmartHome.DEVICE_NAMES[room]
        device = "🟢 ON" if home.devices[room] else "⚫ OFF"
        border = "🟢" if (home.lights[room] or home.devices[room]) else "⚪"
        column.markdown("**%s %s**" % (border, room.title()))
        column.write("Light: %s" % light)
        column.write("%s: %s" % (device_name.title(), device))


def main():
    """Build the page: sidebar, house, mic widget, results, history."""
    st.set_page_config(page_title="Voice-Controlled Smart Home", layout="wide")
    st.title("Voice-Controlled Smart Home Demo")
    init_state()
    model = get_intent_model()

    # Sidebar: engine choice and help.
    st.sidebar.header("Settings")
    engine = st.sidebar.radio("Recognition engine",
                              ["Local Whisper (offline)", "Groq (online fallback)"])
    st.sidebar.write("Confidence threshold: %.2f" % CONFIDENCE_THRESHOLD)
    groq_status = "SET" if groq_key_available() else "NOT SET"
    st.sidebar.write("Groq key status: %s (never shown)" % groq_status)
    with st.sidebar.expander("Setup help"):
        st.write("1. Allow the microphone in the browser address bar.")
        st.write("2. First run needs internet once to download tiny.en (~75 MB).")
        st.write("3. After that, Local Whisper works fully offline.")
        st.write("4. Groq needs GROQ_API_KEY set, e.g. "
                 "`setx GROQ_API_KEY your_key`, then restart.")
        st.write("5. Speak close to the mic, in a quiet room. Single "
                 "commands: 3-5 seconds. Up to %d commands: steady "
                 "speech, max 15 seconds, no long pauses." % MAX_COMMANDS)
    if st.sidebar.button("Check accuracy (groups a-d)"):
        render_accuracy(model)
    render_voice_tally()

    render_house(st.session_state.home)

    # Voice input plus typed fallback.
    st.subheader("Give a command")
    audio = st.audio_input("Record a voice command")
    typed = st.text_input("Or type a command (works without a microphone)",
                          placeholder="turn on the kitchen light")
    col_run, col_reset = st.columns(2)
    run_pressed = col_run.button("Run command")
    if col_reset.button("Reset house"):
        st.session_state.home = SmartHome()
        st.session_state.history = []
        st.session_state.last_result = None
        st.rerun()

    if run_pressed:
        transcript = ""
        if audio is not None:
            # Save browser recording to a temp WAV file for Whisper.
            with tempfile.NamedTemporaryFile(delete=False,
                                             suffix=".wav") as tmp:
                tmp.write(audio.getvalue())
                wav_path = tmp.name
            try:
                if engine.startswith("Groq"):
                    transcript = transcribe_groq(wav_path)
                else:
                    whisper_model = get_whisper_model()
                    transcript = transcribe_local(wav_path, whisper_model)
            except RuntimeError as exc:
                st.error(str(exc))
                transcript = ""
            finally:
                try:
                    os.remove(wav_path)
                except OSError:
                    pass
            if not transcript:
                handle_text("", model, kind="voice")
            else:
                handle_text(transcript, model, kind="voice")
        elif typed.strip():
            handle_text(typed, model, kind="typed")
        else:
            st.warning("Record audio or type a command first.")
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
        st.write("- Max %d commands per sentence; keep recordings under "
                 "15 seconds." % MAX_COMMANDS)


if __name__ == "__main__":
    main()
