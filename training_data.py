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
