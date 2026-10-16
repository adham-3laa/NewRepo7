"""Audio to English text.

Primary engine: local faster-whisper model "tiny.en" on CPU.
It downloads once (~75 MB) on the first run, then works offline.
Optional fallback: Groq Whisper API, only when the user enables it.
"""

import os

# Hint words bias Whisper toward our home vocabulary.
INITIAL_PROMPT = (
    "kitchen bathroom bedroom light door window oven stove "
    "water heater boiler air conditioner aircon turn on turn off "
    "open close lock unlock status"
)


def load_local_model():
    """Load WhisperModel tiny.en for CPU. Downloads once, then cached.

    Raises RuntimeError with a friendly message if the first-run
    download fails (usually no internet on that first run).
    """
    from faster_whisper import WhisperModel
    try:
        return WhisperModel("tiny.en", device="cpu", compute_type="int8")
    except Exception as exc:
        raise RuntimeError(
            "Could not load the local voice model 'tiny.en'. "
            "The first run needs internet once to download about 75 MB. "
            "Connect to the internet and restart. Details: %s" % exc
        )


def transcribe_local(wav_path, model):
    """Transcribe a WAV file with the local model. Returns English text.

    Uses vad_filter to skip silence and an initial prompt with the
    home vocabulary to improve accuracy. Returns "" if nothing heard.
    """
    segments, _info = model.transcribe(
        wav_path,
        language="en",
        vad_filter=True,
        initial_prompt=INITIAL_PROMPT,
    )
    text = "".join(segment.text for segment in segments).strip()
    return text


def groq_key_available():
    """True if a Groq key is set in the environment (never printed)."""
    return bool(os.environ.get("GROQ_API_KEY", "").strip())


def transcribe_groq(wav_path):
    """Transcribe a WAV file with Groq whisper-large-v3-turbo.

    Only call when the user enabled Groq in the sidebar.
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
            )
        return (result.text or "").strip()
    except Exception as exc:
        raise RuntimeError("Groq transcription failed: %s" % exc)
