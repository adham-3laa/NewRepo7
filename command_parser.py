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
# Chosen from measured data (new hand-written sets): at 0.65 all
# single commands still pass (100%), all garbage is rejected (100%),
# and the one wrong action seen at 0.60 ("turn on the bedroom",
# confidence 0.63) falls through to the safe rules instead.
CONFIDENCE_THRESHOLD = 0.65

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


def apply_corrections(text):
    """Fix common tiny.en mishearings. Returns (new_text, [(old, new)]).

    Only the approved list below. Each fires only in a narrow pattern
    with exactly one interpretation. There is deliberately NO
    open->oven correction: a wrong guess must never turn on heating.
    """
    changes = []
    fixed = " %s " % text  # Padded so word-boundary patterns are simple.
    pairs = [
        (" bath room ", " bathroom "),
        (" air con ", " aircon "),  # "air conditioning" never matches this.
        (" log the door ", " lock the door "),
        (" block the door ", " lock the door "),
    ]
    for old, new in pairs:
        if old in fixed:
            fixed = fixed.replace(old, new)
            changes.append((old.strip(), new.strip()))
    # "like" -> "light" only after turn/switch/power + on/off + room name.
    like_pattern = (r"\b(turn|switch|power)\s+(on|off)\s+(the\s+)?"
                    r"(kitchen|bathroom|bedroom)\s+like\b")
    def _like_fix(match):
        verb, direction, _the, room = match.groups()
        return "%s %s the %s light" % (verb, direction, room)
    fixed2, count = re.subn(like_pattern, _like_fix, fixed)
    if count:
        fixed = fixed2
        changes.append(("like", "light"))
    # Standalone "a c" (letters with a space) means the AC.
    fixed2, count = re.subn(r"\ba c\b", "ac", fixed)
    if count:
        fixed = fixed2
        changes.append(("a c", "ac"))
    return (fixed.strip(), changes)


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
                 "shut off", "shut down", "disable", "stop")
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

    # Invalid combinations are never acted on (safety first).
    if ("lock" in t or "unlock" in t) and "window" in t and "door" not in t:
        return "unknown"  # Only the door can be locked.
    if ("open" in t or "close" in t or "shut" in t) and "door" not in t \
            and "window" not in t and _mentions_device(t):
        return "unknown"  # Devices cannot be opened/closed.
    # NOTE: _on_off is not used here: its "on" substring also matches
    # "frONt", which would wrongly reject front-door commands.
    on_off_phrases = ("turn on", "switch on", "power on", "turn off",
                      "switch off", "power off", "shut off", "shut down",
                      " light on", " light off")
    if any(phrase in t for phrase in on_off_phrases) \
            and ("door" in t or "window" in t) \
            and "light" not in t and not _mentions_device(t):
        return "unknown"  # Doors/windows cannot be turned on/off.

    # Status questions.
    if _fuzzy_contains(t, "status") or "what is on" in t or "whats on" in t:
        return "status"
    if "switched on" in t and ("what" in t or "tell me" in t or "show" in t):
        return "status"

    # Global commands first (they contain light/device words too).
    if "everything" in t and _on_off(t) == "off":
        return "everything_off"
    if ("all" in t or "every" in t) and "light" in t and _on_off(t) == "off":
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


def _mentions_device(text):
    """True if the text names any electrical device."""
    names = ("oven", "stove", "water heater", "heater", "boiler",
             "air conditioner", "air conditioning", "aircon", "conditioner",
             "kitchen device", "bathroom device", "bedroom device")
    if any(name in text for name in names):
        return True
    return " ac " in " %s " % text


def _rooms_in(text):
    """Set of room words found in the text, typos allowed.

    Fuzzy matching mirrors rule_fallback (which accepts "kichen" for
    "kitchen"), so the validator never rejects what the rules
    understood. Cutoff 0.8 keeps out lookalikes ("chicken" is 0.77).
    """
    found = set(room for room in _ROOMS if room in text)
    for token in text.split():
        match = difflib.get_close_matches(token, _ROOMS, n=1, cutoff=0.8)
        if match:
            found.add(match[0])
    return found


def _device_hit(text):
    """First (name, room) device found, longest names first. Else None."""
    for name in ("water heater", "air conditioner", "air conditioning",
                 "kitchen device", "bathroom device", "bedroom device",
                 "oven", "stove", "heater", "boiler", "aircon",
                 "conditioner"):
        if name in text:
            for room, names in (("kitchen", ("oven", "stove",
                                             "kitchen device")),
                                ("bathroom", ("water heater", "heater",
                                              "boiler", "bathroom device")),
                                ("bedroom", ("air conditioner",
                                             "air conditioning", "aircon",
                                             "conditioner",
                                             "bedroom device"))):
                if name in names:
                    return (name, room)
    if " ac " in " %s " % text:
        return ("ac", "bedroom")
    return None


def _room_device_name(room):
    """Main device display name for a room (for messages)."""
    return {"kitchen": "oven", "bathroom": "water heater",
            "bedroom": "air conditioner"}[room]


def ambiguity_note(text):
    """Short plain-English note for ambiguous input, or None.

    Covers: room without noun, light/device without room.
    """
    rooms = _rooms_in(text)
    direction = _on_off(text)
    if direction in ("on", "off") and len(rooms) == 1 and \
            "light" not in text and not _mentions_device(text) and \
            "door" not in text and "window" not in text:
        room = next(iter(rooms))
        return "Did you mean the %s light or the %s?" % (
            room, _room_device_name(room))
    if "light" in text and not rooms and not _mentions_device(text):
        return "Which room - kitchen, bathroom, or bedroom?"
    if re.search(r"\bdevice\b", text) and not rooms \
            and _device_hit(text) is None:
        return "Which device - oven, water heater, or air conditioner?"
    return None


def validate_intent(text, intent):
    """Check the intent against the words, ignoring confidence.

    Returns (ok, note). Runs on EVERY result, classifier or rules.
    A missing required word rejects the action, never guessing.
    """
    if intent == "unknown":
        return (True, None)
    rooms = _rooms_in(text)
    if intent.endswith("_light_on") or intent.endswith("_light_off"):
        room = intent.split("_")[0]
        if rooms == set([room]) and "light" in text:
            return (True, None)
        return (False, ambiguity_note(text))
    if intent.endswith("_device_on") or intent.endswith("_device_off"):
        room = intent.split("_")[0]
        hit = _device_hit(text)
        if hit is None or hit[1] != room:
            return (False, ambiguity_note(text))
        if not rooms <= set([room]):
            said = sorted(rooms - set([room]))[0]
            return (False, "The %s is in the %s, not the %s." % (
                hit[0], hit[1], said))
        return (True, None)
    if intent in ("door_open", "door_close"):
        if "door" in text and not rooms:
            return (True, None)
        return (False, None)
    if intent in ("door_lock", "door_unlock"):
        if "door" in text and not rooms and "window" not in text:
            return (True, None)
        return (False, None)
    if intent in ("window_open", "window_close"):
        if "window" in text and not rooms:
            return (True, None)
        return (False, None)
    if intent == "all_lights_off":
        if rooms:
            said = sorted(rooms)[0]
            return (False, "The %s light, or all lights in the house?" % said)
        return (True, None)
    if intent == "everything_off":
        if rooms or _mentions_device(text):
            return (False, "Do you mean everything in the whole house?")
        return (True, None)
    if intent == "status":
        verbs = ("turn on", "switch on", "power on", "turn off",
                 "switch off", "power off", "shut off", "shut down",
                 "open", "close", "shut", "lock", "unlock")
        if any(verb in text for verb in verbs):
            return (False, None)
        return (True, None)
    return (False, None)


def parse_command(text, model):
    """Classifier first, rules second, validator always. Result dict.

    Result keys: text, intent, confidence, source
    (classifier / rule_fallback / validator / unknown), corrections,
    note (short message for ambiguous input, or None).
    validate_intent runs on EVERY non-unknown result, so a confident
    wrong guess is still rejected before anything runs.
    """
    cleaned = normalize(text)
    if not cleaned:
        return {"text": text or "", "intent": "unknown",
                "confidence": 0.0, "source": "unknown", "corrections": [],
                "note": None}
    fixed, corrections = apply_corrections(cleaned)
    note = None
    intent, confidence = predict_intent(model, fixed)
    if confidence >= CONFIDENCE_THRESHOLD:
        source = "classifier"
    else:
        intent = rule_fallback(fixed)
        source = "rule_fallback"
    if intent != "unknown":
        ok, guard_note = validate_intent(fixed, intent)
        if not ok:
            intent = "unknown"
            source = "validator"
            note = guard_note
    if intent == "unknown" and note is None:
        note = ambiguity_note(fixed)
    return {"text": fixed, "intent": intent, "confidence": confidence,
            "source": source, "corrections": corrections, "note": note}


def evaluate_with_rules(model, rows):
    """Score parse_command on (sentence, expected) rows. Groups (b)/(d).

    Expected "SAFE" means the result must be unknown (no action).
    Returns (accuracy, details) with (sentence, expected, intent, conf).
    """
    correct = 0
    details = []
    for sentence, expected in rows:
        result = parse_command(sentence, model)
        if expected == "SAFE":
            good = result["intent"] == "unknown"
        else:
            good = result["intent"] == expected
        if good:
            correct += 1
        details.append((sentence, expected, result["intent"],
                        result["confidence"]))
    return (correct / len(rows) if rows else 0.0, details)
