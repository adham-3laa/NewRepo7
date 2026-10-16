"""Handle MORE THAN ONE command in a single sentence.

Pipeline: normalize speech mistakes ONCE on the whole transcript ->
expand shared words -> split on connectors -> complete each piece
(verb inheritance, exactly one interpretation or reject) -> parse
each piece with the existing parse_command -> run in order.
One refused piece never stops the others. Guessing is never allowed.
"""

import re

from command_parser import apply_corrections, normalize, parse_command
from home import SmartHome
from training_data import get_multi_test

# Maximum commands run per sentence (approved).
MAX_COMMANDS = 4

# Longest first so "and then" splits once, not twice.
CONNECTORS = ("and then", "after that", "followed by", "then",
              "also", "plus", "and")

_PRONOUNS = ("it", "this", "that", "these", "those")
_ROOMS = ("kitchen", "bathroom", "bedroom")
_DEVICE_ROOM = {"oven": "kitchen", "stove": "kitchen",
                "water heater": "bathroom", "heater": "bathroom",
                "boiler": "bathroom", "air conditioner": "bedroom",
                "air conditioning": "bedroom", "ac": "bedroom",
                "aircon": "bedroom", "conditioner": "bedroom"}


def expand_shared(text):
    """Duplicate a shared verb+noun: "turn on the kitchen and
    bedroom lights" -> two full commands. Returns expanded text."""
    pattern = (r"\b(turn|switch|power)\s+(on|off)\s+the\s+"
               r"(kitchen|bathroom|bedroom)\s+and\s+"
               r"(kitchen|bathroom|bedroom)\s+lights?\b")
    return re.sub(pattern, r"\1 \2 the \3 lights and \1 \2 the \4 lights",
                  text)


def split_pieces(text):
    """Cut normalized text into pieces at connector words."""
    lowered = text.lower().replace(",", " and ")
    lowered = normalize(expand_shared(lowered))
    pattern = r"\b(%s)\b" % "|".join(CONNECTORS)
    parts = [part.strip() for part in re.split(pattern, lowered)]
    skip = set(CONNECTORS)
    return [part for part in parts if part and part not in skip]


def _detect_verb(piece):
    """Return the action word of a piece, or None if it has none."""
    if "unlock" in piece:
        return "unlock"
    if "lock" in piece:
        return "lock"
    if "everything" in piece and "off" in piece:
        return "everything_off"
    if ("all" in piece or "every" in piece) and "light" in piece \
            and "off" in piece:
        return "all_lights_off"
    if "status" in piece or "what is on" in piece or "whats on" in piece:
        return "status"
    words = set(piece.split())
    for verb in ("open", "close", "shut"):
        if verb in words:
            return verb
    for verb in ("turn on", "switch on", "power on"):
        if verb in piece:
            return "on"
    for verb in ("turn off", "switch off", "power off", "shut off",
                 "shut down"):
        if verb in piece:
            return "off"
    if " light on" in piece or piece.endswith(" light on"):
        return "on"
    if " light off" in piece or piece.endswith(" light off"):
        return "off"
    return None


def _mentions_target(piece):
    """True if the piece names something to act on."""
    if "door" in piece or "window" in piece or "light" in piece:
        return True
    if "everything" in piece or "all" in piece or "every" in piece:
        return True
    if "status" in piece or "what is on" in piece or "whats on" in piece:
        return True
    return any(name in piece for name in _DEVICE_ROOM)


def complete_piece(piece, prev_verb):
    """Fill a missing verb from the previous piece.

    Returns (full_text_or_None, error_or_None, corrections). STT
    corrections apply first, so "log the door" is seen as a lock
    command. Error pieces run nothing: ambiguous and invalid
    combinations are refused safely, never guessed.
    """
    fixed, corrections = apply_corrections(piece)
    verb = _detect_verb(fixed)
    used_verb = verb or prev_verb
    if verb in ("status", "all_lights_off", "everything_off"):
        return (fixed, None, corrections)
    words = set(fixed.split())
    if used_verb is None:
        if words & set(_PRONOUNS):
            return (None, "I do not know what '%s' refers to. "
                           "Please name the device." % piece,
                    corrections)
        return (None, "I could not find an action in '%s'." % piece,
                corrections)
    # Invalid combinations: refused, never guessed.
    if used_verb in ("lock", "unlock") and "window" in fixed \
            and "door" not in fixed:
        return (None, "The window cannot be locked. "
                       "Only the front door can be locked.",
                corrections)
    if used_verb in ("open", "close", "shut"):
        for name in _DEVICE_ROOM:
            if name in fixed:
                return (None, "The %s cannot be opened or closed." % name,
                        corrections)
    if used_verb in ("on", "off") and ("door" in fixed or "window" in fixed) \
            and "light" not in fixed and not any(
                name in fixed for name in _DEVICE_ROOM):
        return (None, "The door and window cannot be turned on or off.",
                corrections)
    # Pronouns are never resolved.
    if not _mentions_target(fixed) or words & set(_PRONOUNS):
        if words & set(_PRONOUNS):
            return (None, "I do not know what '%s' refers to. "
                           "Please name the device." % piece,
                    corrections)
        if any(room in fixed for room in _ROOMS) and used_verb in ("on", "off"):
            room = next(room for room in _ROOMS if room in fixed)
            device = {"kitchen": "oven", "bathroom": "water heater",
                      "bedroom": "air conditioner"}[room]
            return (None, "I could not tell whether you meant the %s light "
                           "or the %s." % (room, device),
                    corrections)
        return (None, "I did not understand '%s'." % piece, corrections)
    if verb is not None:
        return (fixed, None, corrections)
    # Rebuild a full command from the inherited verb.
    if used_verb in ("on", "off"):
        return ("turn %s %s" % (used_verb, fixed), None, corrections)
    return ("%s %s" % (used_verb, fixed), None, corrections)


def parse_multi(text, model):
    """Split text into pieces and parse each. Returns a result dict.

    Keys: transcript (what was heard, normalized), corrected (the
    transcription after speech-mistake repair), corrections (the
    applied fixes), pieces (list of piece dicts), limited (True when
    more than MAX_COMMANDS were found and extras were dropped).
    """
    transcript = normalize(text)
    if not transcript:
        empty = parse_command("", model)
        return {"transcript": "", "corrected": "", "corrections": [],
                "pieces": [{
                    "piece": "", "full": "", "intent": "unknown",
                    "confidence": 0.0, "source": "unknown", "corrections": [],
                    "note": None, "error": None}], "limited": False}
    # One normalization pass for the whole sentence, before splitting
    # and before shared-word expansion, so both see the corrected
    # rooms/devices. Commas become " and " first, so separators and
    # punctuation survive exactly as before.
    lowered = (text or "").lower().replace(",", " and ")
    corrected, corrections = apply_corrections(normalize(lowered))
    # "heater, and open" becomes "heater and and open"; collapse the
    # duplicate connector for display (splitting drops empty pieces).
    corrected = re.sub(r"\band(?:\s+and\b)+", "and", corrected)
    pieces = split_pieces(corrected)
    limited = len(pieces) > MAX_COMMANDS
    pieces = pieces[:MAX_COMMANDS]
    results = []
    prev_verb = None
    for piece in pieces:
        full, error, piece_corr = complete_piece(piece, prev_verb)
        verb = _detect_verb(piece)
        if verb not in (None, "status", "all_lights_off", "everything_off"):
            prev_verb = verb
        if error is not None:
            results.append({"piece": piece, "full": full or "",
                            "intent": "unknown", "confidence": 0.0,
                            "source": "blocked", "corrections": piece_corr,
                            "note": None, "error": error})
            continue
        parsed = parse_command(full, model)
        parsed["piece"] = piece
        parsed["full"] = full
        parsed["error"] = None
        parsed["corrections"] = piece_corr + parsed["corrections"]
        results.append(parsed)
    return {"transcript": transcript, "corrected": corrected,
            "corrections": corrections, "pieces": results,
            "limited": limited}


def execute_intent(home, intent):
    """Run one intent on the house. Returns (message, changed)."""
    if intent == "kitchen_light_on":
        return home.turn_light_on("kitchen")
    if intent == "kitchen_light_off":
        return home.turn_light_off("kitchen")
    if intent == "bathroom_light_on":
        return home.turn_light_on("bathroom")
    if intent == "bathroom_light_off":
        return home.turn_light_off("bathroom")
    if intent == "bedroom_light_on":
        return home.turn_light_on("bedroom")
    if intent == "bedroom_light_off":
        return home.turn_light_off("bedroom")
    if intent == "kitchen_device_on":
        return home.turn_device_on("kitchen")
    if intent == "kitchen_device_off":
        return home.turn_device_off("kitchen")
    if intent == "bathroom_device_on":
        return home.turn_device_on("bathroom")
    if intent == "bathroom_device_off":
        return home.turn_device_off("bathroom")
    if intent == "bedroom_device_on":
        return home.turn_device_on("bedroom")
    if intent == "bedroom_device_off":
        return home.turn_device_off("bedroom")
    if intent == "door_open":
        return home.open_door()
    if intent == "door_close":
        return home.close_door()
    if intent == "door_lock":
        return home.lock_door()
    if intent == "door_unlock":
        return home.unlock_door()
    if intent == "window_open":
        return home.open_window()
    if intent == "window_close":
        return home.close_window()
    if intent == "all_lights_off":
        return home.all_lights_off()
    if intent == "everything_off":
        return home.everything_off()
    if intent == "status":
        return (home.status_text(), True)
    return ("Sorry, I did not understand that. "
            "Try 'turn on the kitchen light'.", False)


def execute_multi(home, parsed):
    """Run parsed pieces in order. Returns a list of result dicts.

    Each has piece, intent, confidence, source, corrections, message
    and tag: success / already done / refused / not understood.
    A refused piece never stops the others.
    """
    outcomes = []
    for item in parsed["pieces"]:
        if item["error"] is not None:
            outcomes.append(dict(item, message=item["error"],
                                 tag="refused", changed_state=False))
            continue
        if item["intent"] == "unknown" and item.get("note"):
            outcomes.append(dict(item, message=item["note"],
                                 tag="not understood",
                                 changed_state=False))
            continue
        before = _snapshot(home)
        message, changed = execute_intent(home, item["intent"])
        if item["intent"] == "unknown":
            tag = "not understood"
        elif item["intent"] == "status":
            tag = "success"
        elif changed:
            tag = "success"
        elif "already" in message:
            tag = "already done"
        else:
            tag = "refused"
        outcomes.append(dict(item, message=message, tag=tag,
                             changed_state=_snapshot(home) != before))
    if parsed["limited"]:
        outcomes.append({"piece": "...", "intent": "unknown",
                         "confidence": 0.0, "source": "limit",
                         "corrections": [], "note": None, "error": None,
                         "message": "I ran the first %d commands only. "
                                    "Please say the rest separately."
                                    % MAX_COMMANDS, "tag": "refused",
                         "changed_state": False})
    return outcomes


def _snapshot(home):
    """Frozen copy of the house state for safety checks."""
    return (home.door_open, home.door_locked, home.window_open,
            tuple(home.lights[r] for r in ("kitchen", "bathroom", "bedroom")),
            tuple(home.devices[r] for r in ("kitchen", "bathroom", "bedroom")))


def score_multi_test(model):
    """Score group (c) on fresh houses. Returns (cmd_acc, sent_acc, rows).

    A sentence is correct only if ALL its commands are correct.
    "SAFE" means no state change and no real intent. "LIMIT" means the
    limit notice fired. Each row: (sentence, per-piece OK list, sent OK).
    """
    cmd_ok = 0
    cmd_total = 0
    sent_ok = 0
    rows = []
    for sentence, expected in get_multi_test():
        home = SmartHome()
        parsed = parse_multi(sentence, model)
        outcomes = execute_multi(home, parsed)
        piece_ok = []
        for outcome, exp in zip(outcomes, expected):
            if exp == "SAFE":
                # Safe = refused/rejected with no state change. The intent
                # may be real (e.g. door_lock refused because the door is
                # open) as long as the home layer said no.
                good = outcome["tag"] in ("refused", "not understood") and \
                    not outcome["changed_state"] and \
                    not outcome["changed_state"]
            elif exp == "LIMIT":
                good = outcome["tag"] == "refused" and \
                    "first %d" % MAX_COMMANDS in outcome["message"]
            else:
                good = outcome["intent"] == exp and \
                    outcome["tag"] in ("success", "already done")
            piece_ok.append(good)
        sent_good = len(outcomes) == len(expected) and all(piece_ok)
        cmd_ok += sum(piece_ok)
        cmd_total += len(expected)
        sent_ok += 1 if sent_good else 0
        rows.append((sentence, piece_ok, sent_good))
    return (cmd_ok / cmd_total if cmd_total else 0.0,
            sent_ok / len(rows) if rows else 0.0, rows)
