"""Audio to English text.

Primary engine: local faster-whisper "tiny.en" on CPU (~75 MB
download once, then offline). Benchmarked against "base.en" on short
smart-home commands: equal accuracy, base.en ~2x slower, so tiny.en
stays the default. Set the environment variable WHISPER_MODEL to
"base.en" (~145 MB) or "small.en" (~480 MB) for a bigger model.
Optional fallback: Groq Whisper API, only when the user enables it.
"""

import os

# Hint words bias Whisper toward our home vocabulary.
# Connector words (and, then, ...) are included so multi-command
# sentences keep the words the splitter needs.
INITIAL_PROMPT = (
    "kitchen bathroom bedroom light oven water heater air conditioner "
    "front door window open close lock unlock turn on turn off "
    "all lights off everything off status and then also plus after that"
)

# Friendly download sizes for the error/UI messages.
MODEL_SIZES_MB = {
    "tiny.en": 75,
    "base.en": 145,
    "small.en": 480,
}
DEFAULT_LOCAL_MODEL = "tiny.en"


def local_model_name():
    """Model name for the local engine. WHISPER_MODEL overrides."""
    name = (os.environ.get("WHISPER_MODEL") or "").strip()
    return name or DEFAULT_LOCAL_MODEL


def local_model_size_text(name=None):
    """Human text for the first-run download, e.g. 'about 145 MB'."""
    size = MODEL_SIZES_MB.get(name or local_model_name())
    return "about %d MB" % size if size else "a few hundred MB"


def load_local_model(name=None):
    """Load a WhisperModel on CPU. Downloads once, then cached.

    Raises RuntimeError with a friendly message if the first-run
    download fails (usually no internet on that first run).
    """
    name = name or local_model_name()
    from faster_whisper import WhisperModel
    try:
        return WhisperModel(name, device="cpu", compute_type="int8")
    except Exception as exc:
        raise RuntimeError(
            "Could not load the local voice model '%s'. "
            "The first run needs internet once to download %s. "
            "Connect to the internet and restart. Tip: "
            "set WHISPER_MODEL=base.en (or small.en) for a larger, "
            "more accurate model. "
            "Details: %s" % (name, local_model_size_text(name), exc)
        )


def transcribe_local(wav_path, model):
    """Transcribe a WAV file with the local model. Returns English text.

    Tuned for short English smart-home commands: greedy-free beam
    search, no cross-segment conditioning (each recording stands
    alone), VAD to skip silence, and both the initial prompt and
    hotwords biasing toward the house vocabulary. Returns "" if
    nothing was heard.
    """
    segments, _info = model.transcribe(
        wav_path,
        language="en",
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=True,
        initial_prompt=INITIAL_PROMPT,
        hotwords=INITIAL_PROMPT,
    )
    text = "".join(segment.text for segment in segments).strip()
    return text


def groq_key_available():
    """True if a Groq key is set in the environment (never printed)."""
    return bool(os.environ.get("GROQ_API_KEY", "").strip())


def transcribe_groq(wav_path):
    """Transcribe a WAV file with Groq whisper-large-v3-turbo.

    Only call when the user enabled Groq in the sidebar.
    Uses the same vocabulary prompt as the local engine and
    temperature 0.0 for the most repeatable short-command output.
    Raises RuntimeError if the key is missing or the call fails.
    """
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at console.groq.com, "
            "then set it (Windows: setx GROQ_API_KEY your_key) and restart."
        )
    from groq import Groq
    try:
        client = Groq(api_key=api_key)
        with open(wav_path, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=audio_file,
                language="en",
                prompt=INITIAL_PROMPT,
                temperature=0.0,
                response_format="json",
            )
        return (result.text or "").strip()
    except Exception as exc:
        raise RuntimeError("Groq transcription failed: %s" % exc)
