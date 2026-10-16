"""Streamlit web interface for the voice-controlled smart home."""

import os
import tempfile

import streamlit as st

from command_parser import CONFIDENCE_THRESHOLD, parse_command
from home import SmartHome
from intent_model import evaluate_accuracy, load_or_train
from speech import (
    groq_key_available,
    load_local_model,
    transcribe_groq,
    transcribe_local,
)


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
    if "home" not in st.session_state:
        st.session_state.home = SmartHome()
    if "history" not in st.session_state:
        st.session_state.history = []  # List of (heard, intent, message).
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def apply_intent(home, intent):
    """Run one intent on the house. Returns a message. Never crashes."""
    if intent == "kitchen_light_on":
        message, _ = home.turn_light_on("kitchen")
    elif intent == "kitchen_light_off":
        message, _ = home.turn_light_off("kitchen")
    elif intent == "bathroom_light_on":
        message, _ = home.turn_light_on("bathroom")
    elif intent == "bathroom_light_off":
        message, _ = home.turn_light_off("bathroom")
    elif intent == "bedroom_light_on":
        message, _ = home.turn_light_on("bedroom")
    elif intent == "bedroom_light_off":
        message, _ = home.turn_light_off("bedroom")
    elif intent == "kitchen_device_on":
        message, _ = home.turn_device_on("kitchen")
    elif intent == "kitchen_device_off":
        message, _ = home.turn_device_off("kitchen")
    elif intent == "bathroom_device_on":
        message, _ = home.turn_device_on("bathroom")
    elif intent == "bathroom_device_off":
        message, _ = home.turn_device_off("bathroom")
    elif intent == "bedroom_device_on":
        message, _ = home.turn_device_on("bedroom")
    elif intent == "bedroom_device_off":
        message, _ = home.turn_device_off("bedroom")
    elif intent == "door_open":
        message, _ = home.open_door()
    elif intent == "door_close":
        message, _ = home.close_door()
    elif intent == "door_lock":
        message, _ = home.lock_door()
    elif intent == "door_unlock":
        message, _ = home.unlock_door()
    elif intent == "window_open":
        message, _ = home.open_window()
    elif intent == "window_close":
        message, _ = home.close_window()
    elif intent == "all_lights_off":
        message, _ = home.all_lights_off()
    elif intent == "everything_off":
        message, _ = home.everything_off()
    elif intent == "status":
        message = home.status_text()
    else:
        message = ("Sorry, I did not understand that. "
                   "Try 'turn on the kitchen light'.")
    return message


def handle_text(text, model):
    """Parse text, update the house, store transcript and history."""
    if not (text or "").strip():
        st.session_state.last_result = ("", "unknown", 0.0, "unknown",
                                        "I heard nothing. Please speak closer "
                                        "or type a command.")
        return
    result = parse_command(text, model)
    message = apply_intent(st.session_state.home, result["intent"])
    st.session_state.last_result = (result["text"], result["intent"],
                                    result["confidence"], result["source"],
                                    message)
    st.session_state.history.append(
        (result["text"], result["intent"], message))
    st.session_state.history = st.session_state.history[-10:]


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
        st.write("5. Speak 3-5 seconds, close to the mic, in a quiet room.")
    if st.sidebar.button("Check classifier accuracy"):
        accuracy, details = evaluate_accuracy(model)
        st.sidebar.write("Accuracy on unseen sentences: %.0f%%" % (accuracy * 100))
        for sentence, expected, predicted, conf in details[:10]:
            mark = "OK" if expected == predicted else "MISS"
            st.sidebar.write("%s '%s' -> %s (%.2f)" % (
                mark, sentence, predicted, conf))

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
                handle_text("", model)
            else:
                handle_text(transcript, model)
        elif typed.strip():
            handle_text(typed, model)
        else:
            st.warning("Record audio or type a command first.")
        st.rerun()

    # Show what was heard, the intent with confidence, and history.
    if st.session_state.last_result is not None:
        heard, intent, confidence, source, message = st.session_state.last_result
        st.subheader("Result")
        st.write("Heard: '%s'" % heard)
        st.write("Intent: %s (confidence %.2f, via %s, threshold %.2f)" % (
            intent, confidence, source, CONFIDENCE_THRESHOLD))
        st.info(message)

    if st.session_state.history:
        st.subheader("History (last %d)" % len(st.session_state.history))
        for heard, intent, message in reversed(st.session_state.history):
            st.write("- '%s' -> %s: %s" % (heard, intent, message))

    with st.expander("Example commands"):
        st.write("- turn on the kitchen light / switch off the bedroom light")
        st.write("- turn on the oven / turn off the air conditioner")
        st.write("- turn on the bedroom device")
        st.write("- open the door / close the window / lock the door")
        st.write("- turn off all lights / turn everything off")
        st.write("- what is the status of the house")


if __name__ == "__main__":
    main()
