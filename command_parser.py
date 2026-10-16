"""Map English text to a house action.

Two layers:
1. The scikit-learn classifier predicts (intent, confidence).
2. If confidence is below CONFIDENCE_THRESHOLD, plain keyword
   rules plus difflib fuzzy matching decide instead, so a weak
   guess never triggers a wrong action.
"""

import difflib
import re

from intent_model import predict_intent

# Single tunable constant. Shown in the UI so it can be tuned.
CONFIDENCE_THRESHOLD = 0.60

_ROOMS = ("kitchen", "bathroom", "bedroom")

_KITCHEN_DEVICES = ("oven", "stove", "kitchen device")
_BATHROOM_DEVICES = ("water heater", "heater", "boiler", "bathroom device")
_BEDROOM_DEVICES = ("air conditioner", "air conditioning", "ac",
                    "aircon", "conditioner", "bedroom device")


def normalize(text):
    """Lowercase, remove punctuation, collapse spaces."""
    cleaned = (text or "").lower()
    cleaned = re.sub(r"[^a-z0-9 ]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _fuzzy_contains(text, *words, cutoff=0.8):
    """True if any word appears in text, allowing small typos."""
    tokens = text.split()
    for target in words:
        if target in text:
            return True
        if " " in target:
            continue  # Multi-word phrases need exact match.
        matches = difflib.get_close_matches(target, tokens, n=1, cutoff=cutoff)
        if matches:
            return True
    return False


def _on_off(text):
    """Return 'on', 'off', or None based on keywords in text."""
    on_words = ("turn on", "switch on", "power on", " light on", "enable", "start")
    off_words = ("turn off", "switch off", "power off", " light off",
                 "shut off", "disable", "stop")
    # Check multi-word phrases first for accuracy.
    if any(phrase in text for phrase in off_words):
        return "off"
    if any(phrase in text for phrase in on_words):
        return "on"
    # Single-word fallback with fuzzy matching.
    if _fuzzy_contains(text, "off"):
        return "off"
    if _fuzzy_contains(text, "on"):
        return "on"
    return None


def rule_fallback(text):
    """Keyword rules. Returns an intent string or 'unknown'."""
    t = normalize(text)
    if not t:
        return "unknown"

    # Status questions.
    if _fuzzy_contains(t, "status") or "what is on" in t or "whats on" in t:
        return "status"
    if "switched on" in t and ("what" in t or "tell me" in t or "show" in t):
        return "status"

    # Global commands first (they contain light/device words too).
    if "everything" in t and _on_off(t) == "off":
        return "everything_off"
    if "all" in t and "light" in t and _on_off(t) == "off":
        return "all_lights_off"

    # Door: exact match only (no fuzzy here: "lock" fuzzy-matches
    # "unlock" and they are opposite actions). Unlock first because
    # "unlock" contains "lock".
    if "unlock" in t:
        return "door_unlock"
    if "lock" in t:
        return "door_lock"
    if "door" in t or _fuzzy_contains(t, "door"):
        if _fuzzy_contains(t, "open"):
            return "door_open"
        if _fuzzy_contains(t, "close", "shut"):
            return "door_close"

    # Window.
    if "window" in t or _fuzzy_contains(t, "window"):
        if _fuzzy_contains(t, "open"):
            return "window_open"
        if _fuzzy_contains(t, "close", "shut"):
            return "window_close"

    # Room lights.
    for room in _ROOMS:
        if room in t or _fuzzy_contains(t, room):
            if "light" in t or _fuzzy_contains(t, "light"):
                direction = _on_off(t)
                if direction == "on":
                    return "%s_light_on" % room
                if direction == "off":
                    return "%s_light_off" % room

    # Devices by name (also catches generic "kitchen device" etc.).
    direction = _on_off(t)
    if direction in ("on", "off"):
        suffix = "device_%s" % direction
        for name in _KITCHEN_DEVICES:
            if name in t:
                return "kitchen_" + suffix
        for name in _BATHROOM_DEVICES:
            if name in t:
                return "bathroom_" + suffix
        for name in _BEDROOM_DEVICES:
            if name in t:
                return "bedroom_" + suffix
        # Fuzzy single-word device names (oven, stove, heater, ...).
        for name in ("oven", "stove", "heater", "boiler",
                     "aircon", "conditioner"):
            if _fuzzy_contains(t, name):
                if name in ("oven", "stove"):
                    return "kitchen_" + suffix
                if name in ("heater", "boiler"):
                    return "bathroom_" + suffix
                return "bedroom_" + suffix
        if "ac" in t.split():
            return "bedroom_" + suffix

    return "unknown"


def parse_command(text, model):
    """Classifier first, rules second. Returns a result dict.

    Result keys: text, intent, confidence, source
    (classifier / rule_fallback / unknown).
    A wrong classifier guess below threshold never triggers an action.
    """
    cleaned = normalize(text)
    if not cleaned:
        return {"text": text or "", "intent": "unknown",
                "confidence": 0.0, "source": "unknown"}
    intent, confidence = predict_intent(model, cleaned)
    if confidence >= CONFIDENCE_THRESHOLD:
        return {"text": cleaned, "intent": intent,
                "confidence": confidence, "source": "classifier"}
    rule_intent = rule_fallback(cleaned)
    if rule_intent != "unknown":
        return {"text": cleaned, "intent": rule_intent,
                "confidence": confidence, "source": "rule_fallback"}
    return {"text": cleaned, "intent": "unknown",
            "confidence": confidence, "source": "unknown"}
