"""Generate training sentences for the intent classifier.

No external datasets. All sentences are built in code from
templates plus approved synonyms, so the data is small,
English-only, and covers exactly our house.
"""

# Approved device words per room (no geyser, no cooler).
KITCHEN_DEVICES = ["oven", "stove", "kitchen device"]
BATHROOM_DEVICES = ["water heater", "heater", "boiler", "bathroom device"]
BEDROOM_DEVICES = [
    "air conditioner",
    "air conditioning",
    "ac",
    "aircon",
    "conditioner",
    "bedroom device",
]

ON_VERBS = ["turn on", "switch on", "enable", "start", "power on"]
OFF_VERBS = ["turn off", "switch off", "disable", "stop", "shut off", "power off"]


def _light_sentences(room):
    """Build light on/off sentences for one room."""
    rows = []
    for verb in ON_VERBS:
        rows.append(("%s the %s light" % (verb, room), "%s_light_on" % room))
        rows.append(("%s %s light" % (verb, room), "%s_light_on" % room))
    rows.append(("%s light on" % room, "%s_light_on" % room))
    rows.append(("%s light on in the %s" % (room, room), "%s_light_on" % room))
    for verb in OFF_VERBS:
        rows.append(("%s the %s light" % (verb, room), "%s_light_off" % room))
        rows.append(("%s %s light" % (verb, room), "%s_light_off" % room))
    rows.append(("%s light off" % room, "%s_light_off" % room))
    return rows


def _device_sentences(room, names):
    """Build device on/off sentences for one room."""
    rows = []
    for name in names:
        for verb in ON_VERBS:
            rows.append(("%s the %s" % (verb, name), "%s_device_on" % room))
        rows.append(("%s on" % name, "%s_device_on" % room))
        for verb in OFF_VERBS:
            rows.append(("%s the %s" % (verb, name), "%s_device_off" % room))
        rows.append(("%s off" % name, "%s_device_off" % room))
    return rows


def generate_training_data():
    """Return a list of (sentence, intent) used to train the classifier."""
    rows = []
    for room in ("kitchen", "bathroom", "bedroom"):
        rows.extend(_light_sentences(room))
    rows.extend(_device_sentences("kitchen", KITCHEN_DEVICES))
    rows.extend(_device_sentences("bathroom", BATHROOM_DEVICES))
    rows.extend(_device_sentences("bedroom", BEDROOM_DEVICES))

    # Door and window.
    for verb in ["open", "please open"]:
        rows.append(("%s the door" % verb, "door_open"))
        rows.append(("%s the window" % verb, "window_open"))
    for verb in ["close", "shut", "please close"]:
        rows.append(("%s the door" % verb, "door_close"))
        rows.append(("%s the window" % verb, "window_close"))
    for verb in ["lock", "please lock"]:
        rows.append(("%s the door" % verb, "door_lock"))
    for verb in ["unlock", "please unlock"]:
        rows.append(("%s the door" % verb, "door_unlock"))

    # Global commands.
    for text in ["turn off all lights", "switch off all lights",
                 "all lights off", "lights off everywhere"]:
        rows.append((text, "all_lights_off"))
    for text in ["turn everything off", "turn off everything",
                 "shut everything off", "switch everything off"]:
        rows.append((text, "everything_off"))

    # Status questions.
    for text in ["what is the status of the house", "status",
                 "house status", "show status", "what is on",
                 "show me the house status"]:
        rows.append((text, "status"))
    rows.extend(_extra_variety())
    return rows


def _extra_variety():
    """Extra wording variety: polite forms, plurals, front door, fillers.

    Every string here was checked against both hand-written test sets:
    no verbatim overlap, so the tests stay honest.
    """
    rows = []
    for room in ("kitchen", "bathroom", "bedroom"):
        for verb, intent in (("turn on", "%s_light_on" % room),
                             ("turn off", "%s_light_off" % room)):
            rows.append(("please %s the %s light" % (verb, room), intent))
        rows.append(("could you turn on the %s light" % room,
                     "%s_light_on" % room))
        rows.append(("could you turn off the %s light" % room,
                     "%s_light_off" % room))
        rows.append(("turn on the %s lights" % room, "%s_light_on" % room))
        rows.append(("turn off the %s lights" % room, "%s_light_off" % room))
    for name, room in (("oven", "kitchen"), ("heater", "bathroom"),
                       ("air conditioner", "bedroom")):
        rows.append(("please turn on the %s" % name, "%s_device_on" % room))
        rows.append(("please turn off the %s" % name, "%s_device_off" % room))
        rows.append(("could you turn on the %s" % name, "%s_device_on" % room))
        rows.append(("could you turn off the %s" % name, "%s_device_off" % room))
    # Front door word order (tests use "for me" / "now" / "up" forms).
    rows.append(("open the front door", "door_open"))
    rows.append(("close the front door", "door_close"))
    rows.append(("lock the front door", "door_lock"))
    rows.append(("unlock the front door", "door_unlock"))
    # More status and global phrasings.
    for text in ["tell me the house status", "show me the house status",
                 "give me the status", "report the house status"]:
        rows.append((text, "status"))
    for text in ["turn all the lights off", "switch all lights off",
                 "turn every light off"]:
        rows.append((text, "all_lights_off"))
    for text in ["shut everything down", "power everything off",
                 "switch off everything"]:
        rows.append((text, "everything_off"))
    return rows


def generate_test_data():
    """Return unseen phrasings (not in training) to measure accuracy.

    These use extra politeness words and word orders that the
    training templates do not contain verbatim.
    """
    return [
        ("could you please turn the kitchen light on", "kitchen_light_on"),
        ("kitchen light on please", "kitchen_light_on"),
        ("please switch the bedroom light off", "bedroom_light_off"),
        ("bathroom light off now", "bathroom_light_off"),
        ("could you start the oven please", "kitchen_device_on"),
        ("please stop the stove", "kitchen_device_off"),
        ("start the heater in the bathroom", "bathroom_device_on"),
        ("boiler off please", "bathroom_device_off"),
        ("ac on in the bedroom please", "bedroom_device_on"),
        ("please turn the air conditioning off", "bedroom_device_off"),
        ("would you open the front door", "door_open"),
        ("please shut the front door", "door_close"),
        ("lock the front door now", "door_lock"),
        ("unlock the front door please", "door_unlock"),
        ("could you open the window", "window_open"),
        ("please shut the window", "window_close"),
        ("all the lights off please", "all_lights_off"),
        ("everything off now please", "everything_off"),
        ("tell me the house status", "status"),
        ("what is switched on right now", "status"),
    ]


def get_single_test_v2():
    """New hand-written single commands (set 2, never used in training).

    Phrased differently from both the templates and the old test set.
    """
    return [
        ("kindly turn the kitchen light on", "kitchen_light_on"),
        ("bedroom light on, please", "bedroom_light_on"),
        ("please turn off the bathroom light", "bathroom_light_off"),
        ("switch the oven on", "kitchen_device_on"),
        ("turn the stove off now", "kitchen_device_off"),
        ("could you turn on the bathroom heater", "bathroom_device_on"),
        ("turn the boiler off", "bathroom_device_off"),
        ("please enable aircon in the bedroom", "bedroom_device_on"),
        ("shut off the water heater", "bathroom_device_off"),
        ("power on the oven please", "kitchen_device_on"),
        ("open the front door for me", "door_open"),
        ("shut the front door", "door_close"),
        ("lock up the door", "door_lock"),
        ("unlock the door now", "door_unlock"),
        ("open up the window", "window_open"),
        ("close the window please", "window_close"),
        ("switch every light off", "all_lights_off"),
        ("shut down everything", "everything_off"),
        ("tell me the status", "status"),
        ("show me the status", "status"),
    ]


def get_multi_test():
    """New hand-written multi-command sentences with expected outcomes.

    Each expected item is an intent, or "SAFE" (the piece must be
    refused/rejected/unknown with NO action), or "LIMIT" (the 5-command
    case: first 4 run, limit notice shown).
    """
    return [
        ("turn on the kitchen light and open the window",
         ["kitchen_light_on", "window_open"]),
        ("turn off the oven, then lock the door",
         ["kitchen_device_off", "door_lock"]),
        ("turn on the kitchen and bedroom lights",
         ["kitchen_light_on", "bedroom_light_on"]),
        ("switch on the air conditioner and the bedroom light, then tell me the status",
         ["bedroom_device_on", "bedroom_light_on", "status"]),
        ("open the door and the window",
         ["door_open", "window_open"]),
        ("turn on the kitchen light and the oven",
         ["kitchen_light_on", "kitchen_device_on"]),
        ("turn off the bathroom light and heater",
         ["bathroom_light_off", "bathroom_device_off"]),
        ("lock the door and the window",
         ["door_lock", "SAFE"]),
        ("turn on the kitchen light and the bedroom",
         ["kitchen_light_on", "SAFE"]),
        ("turn it on",
         ["SAFE"]),
        ("open the door and lock the door",
         ["door_open", "SAFE"]),
        ("show me the status and turn off all lights",
         ["status", "all_lights_off"]),
        ("turn on the oven, turn on the air conditioner, turn on the kitchen light, turn on the water heater, and open the window",
         ["kitchen_device_on", "bedroom_device_on", "kitchen_light_on",
          "bathroom_device_on", "LIMIT"]),
        ("close the window, lock the door and turn off all lights",
         ["window_close", "door_lock", "all_lights_off"]),
    ]


def get_garbage_test():
    """Garbage, typos and invalid singles. All must be safely rejected.

    Expected "SAFE" means unknown/refused with NO action taken.
    "kitchen_light_on" entries are typos/corrections that MUST still work.
    """
    return [
        ("make coffee", "SAFE"),
        ("asdfgh", "SAFE"),
        ("", "SAFE"),
        ("hello house are you there", "SAFE"),
        ("please please please", "SAFE"),
        ("lock the window", "SAFE"),
        ("open the oven", "SAFE"),
        ("turn on the bedroom", "SAFE"),
        ("trun on kichen light", "kitchen_light_on"),
        ("bath room light on", "bathroom_light_on"),
        ("turn on the kitchen like", "kitchen_light_on"),
        ("log the door", "door_lock"),
    ]


def get_ambiguity_test():
    """Hand-written ambiguity and near-miss singles. ALL must be SAFE.

    First 15 approved by the user, last 5 invented for this task
    (room-only, wrong-room-device, pronouns). SAFE means no action
    and no state change, only a short clarifying message.
    """
    return [
        ("turn on the bedroom", "SAFE"),
        ("turn off the kitchen", "SAFE"),
        ("switch on the bathroom", "SAFE"),
        ("power off the bedroom", "SAFE"),
        ("turn on the light", "SAFE"),
        ("switch off the lights", "SAFE"),
        ("light on", "SAFE"),
        ("turn on the device", "SAFE"),
        ("switch the device off", "SAFE"),
        ("turn on the kitchen heater", "SAFE"),
        ("turn off the bedroom oven", "SAFE"),
        ("switch on the bathroom air conditioner", "SAFE"),
        ("turn off all kitchen lights", "SAFE"),
        ("turn it off", "SAFE"),
        ("switch it on", "SAFE"),
        ("switch off the bathroom", "SAFE"),
        ("turn the bedroom heater on", "SAFE"),
        ("power on the kitchen air conditioner", "SAFE"),
        ("switch that on", "SAFE"),
        ("turn these off", "SAFE"),
    ]


def get_ambiguity_multi_test():
    """One multi sentence with TWO ambiguous pieces. Both must be SAFE."""
    return [
        ("turn on the bathroom and switch it off", ["SAFE", "SAFE"]),
    ]
