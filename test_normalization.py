"""Focused tests for tolerant speech normalization and command parsing.

Run with:
    python -m unittest test_normalization -v

Covers: explicit aliases, conservative fuzzy repair, protected words,
single commands, multi-command utterances, ambiguity/unknown safety,
and regression of all previously supported behavior.
"""

import unittest

from command_parser import (
    CONFIDENCE_THRESHOLD,
    apply_corrections,
    evaluate_with_rules,
    parse_command,
)
from home import SmartHome
from intent_model import evaluate_accuracy, evaluate_single_v2, load_or_train
from multi_command import (
    MAX_COMMANDS,
    execute_multi,
    parse_multi,
    score_multi_test,
)
from training_data import (
    get_ambiguity_multi_test,
    get_ambiguity_test,
    get_garbage_test,
    get_multi_test,
    get_single_test_v2,
    get_stt_variant_test,
)

MODEL = None


def setUpModule():
    """Load the cached intent classifier once for the whole module."""
    global MODEL
    MODEL = load_or_train()


def run_utterance(text):
    """Parse and execute one utterance on a fresh house.

    Returns (home, parsed, outcomes) so tests can check the final house
    state and each individual action outcome.
    """
    home = SmartHome()
    parsed = parse_multi(text, MODEL)
    outcomes = execute_multi(home, parsed)
    return home, parsed, outcomes


class TestCorrections(unittest.TestCase):
    """Unit tests for apply_corrections itself."""

    def check(self, original, expected, expect_record=True):
        fixed, changes = apply_corrections(original)
        self.assertEqual(fixed, expected,
                         "%r -> %r (changes %r)" % (original, fixed, changes))
        if expect_record:
            self.assertTrue(changes, "no correction recorded for %r"
                            % original)

    def test_room_aliases(self):
        self.check("bathrom light on", "bathroom light on")
        self.check("bathrum light on", "bathroom light on")
        self.check("bathroon light on", "bathroom light on")
        self.check("bath room light on", "bathroom light on")
        self.check("bedrom light on", "bedroom light on")
        self.check("bedroon light on", "bedroom light on")
        self.check("bed room light on", "bedroom light on")
        self.check("kichen light on", "kitchen light on")
        self.check("kitchin light on", "kitchen light on")
        self.check("kitcen light on", "kitchen light on")

    def test_badroom_resolves_to_bedroom_only(self):
        # "badroom" is one edit from bedroom, two from bathroom.
        self.check("badroom light on", "bedroom light on")

    def test_device_and_object_aliases(self):
        self.check("turn on the ovem", "turn on the oven")
        self.check("open the windo", "open the window")
        self.check("close the windoe", "close the window")
        self.check("turn on the bedroom lite", "turn on the bedroom light")
        self.check("turn on the water heather", "turn on the water heater")
        self.check("turn off the watter heater", "turn off the water heater")
        self.check("air condition off", "air conditioner off")
        self.check("a c on in the bedroom", "ac on in the bedroom")
        self.check("open the dor", "open the door")

    def test_action_aliases(self):
        self.check("tun on the kitchen light", "turn on the kitchen light")
        self.check("trun on kichen light", "turn on kitchen light")
        self.check("log the door", "lock the door")
        self.check("turn on the kitchen like", "turn on the kitchen light")

    def test_correct_transcript_unchanged(self):
        for text in ("turn on the kitchen light", "open the front door",
                     "what is the status of the house", "all lights off",
                     "turn off the water heater"):
            fixed, changes = apply_corrections(text)
            self.assertEqual(fixed, text)
            self.assertEqual(changes, [])

    def test_protected_words_never_rewritten(self):
        protected = (
            "right", "left", "night", "might", "fight", "tight", "sight",
            "eight", "flight", "burn", "torn", "burnt", "over", "even",
            "look", "clock", "block", "flock", "poor", "four", "hour",
            "stone", "store", "stole", "stave", "statue", "statute",
            "stature", "stratus", "star", "stoop", "shout", "stitch",
            "twitch", "witch", "closet", "woven",
        )
        for word in protected:
            fixed, changes = apply_corrections("turn on the %s" % word)
            self.assertEqual(fixed, "turn on the %s" % word,
                             "protected word was rewritten: %s" % word)
            self.assertEqual(changes, [])

    def test_status_phrase_untouched(self):
        # A regression sentence from the old accuracy set: must not
        # become "...light now".
        self.check("what is switched on right now",
                   "what is switched on right now", expect_record=False)

    def test_unrelated_text_untouched(self):
        for text in ("make coffee", "chicken recipe", "store hours",
                     "hello house are you there", "asdfgh",
                     "the statue is in the garden"):
            fixed, changes = apply_corrections(text)
            self.assertEqual(fixed, text)
            self.assertEqual(changes, [])

    def test_idempotent_and_no_duplicate_records(self):
        fixed, changes = apply_corrections("bathrom light on and turn on "
                                           "the ovem")
        self.assertEqual(changes, [("bathrom", "bathroom"),
                                   ("ovem", "oven")])
        fixed2, changes2 = apply_corrections(fixed)
        self.assertEqual(fixed2, fixed)
        self.assertEqual(changes2, [])

    def test_repeated_typo_recorded_once(self):
        _fixed, changes = apply_corrections(
            "bathrom light on and bathrom light off")
        self.assertEqual(changes, [("bathrom", "bathroom")])


class TestSingleCommands(unittest.TestCase):
    """parse_command behavior on variants and negatives."""

    def intent(self, text):
        return parse_command(text, MODEL)["intent"]

    def test_correct_commands_still_work(self):
        cases = {
            "turn on the kitchen light": "kitchen_light_on",
            "turn off the bedroom light": "bedroom_light_off",
            "switch the oven on": "kitchen_device_on",
            "turn on the air conditioner": "bedroom_device_on",
            "open the front door": "door_open",
            "lock the door": "door_lock",
            "close the window": "window_close",
            "turn off all lights": "all_lights_off",
            "turn everything off": "everything_off",
            "tell me the status": "status",
        }
        for text, expected in cases.items():
            self.assertEqual(self.intent(text), expected, text)

    def test_misspelling_variants_work(self):
        cases = {
            "bathrom light on": "bathroom_light_on",
            "bath room light on": "bathroom_light_on",
            "badroom light on": "bedroom_light_on",
            "kichen light off": "kitchen_light_off",
            "turn on the ovem": "kitchen_device_on",
            "open the windo": "window_open",
            "turn on the bedroom lite": "bedroom_light_on",
            "turn on the water heather": "bathroom_device_on",
        }
        for text, expected in cases.items():
            self.assertEqual(self.intent(text), expected, text)

    def test_extra_short_word_still_understood(self):
        self.assertEqual(self.intent("turn on in kitchen light"),
                         "kitchen_light_on")

    def test_corrections_are_exposed(self):
        result = parse_command("bathrom light on", MODEL)
        self.assertEqual(result["intent"], "bathroom_light_on")
        self.assertEqual(result["text"], "bathroom light on")
        self.assertTrue(result["corrections"])

    def test_ambiguity_and_unknown_stay_safe(self):
        safe = [
            "turn on the bedroom",
            "turn off the kitchen",
            "turn on the light",
            "switch the device off",
            "turn on the kitchen heater",
            "turn off the bedroom oven",
            "turn it off",
            "lock the window",
            "open the oven",
            "make coffee",
            "close the closet",
            "turn on the store light",
            "the star light on",
        ]
        for text in safe:
            self.assertEqual(self.intent(text), "unknown", text)

    def test_threshold_unchanged(self):
        self.assertEqual(CONFIDENCE_THRESHOLD, 0.65)


class TestMultiCommands(unittest.TestCase):
    """parse_multi + execute_multi on fresh houses."""

    def test_typo_multi_runs_all_pieces(self):
        home, parsed, outcomes = run_utterance(
            "bathrom light on and turn on the ovem")
        self.assertEqual([o["intent"] for o in outcomes],
                         ["bathroom_light_on", "kitchen_device_on"])
        self.assertEqual([o["tag"] for o in outcomes], ["success", "success"])
        self.assertTrue(home.lights["bathroom"])
        self.assertTrue(home.devices["kitchen"])
        self.assertEqual(parsed["corrections"],
                         [("bathrom", "bathroom"), ("ovem", "oven")])

    def test_shared_words_expand_with_typo(self):
        home, parsed, outcomes = run_utterance(
            "turn on the kitchen and bathrom lights")
        self.assertEqual(len(outcomes), 2)
        self.assertEqual([o["intent"] for o in outcomes],
                         ["kitchen_light_on", "bathroom_light_on"])
        self.assertTrue(all(o["tag"] == "success" for o in outcomes))
        self.assertTrue(home.lights["kitchen"])
        self.assertTrue(home.lights["bathroom"])

    def test_typo_door_and_window_multi(self):
        home, _parsed, outcomes = run_utterance(
            "open the windo then open the door")
        self.assertEqual([o["intent"] for o in outcomes],
                         ["window_open", "door_open"])
        self.assertTrue(all(o["tag"] == "success" for o in outcomes))
        self.assertTrue(home.window_open)
        self.assertTrue(home.door_open)

    def test_ambiguous_piece_still_refused(self):
        home, _parsed, outcomes = run_utterance(
            "turn on the kichen light and the bedrom")
        self.assertEqual(outcomes[0]["intent"], "kitchen_light_on")
        self.assertEqual(outcomes[0]["tag"], "success")
        self.assertEqual(outcomes[1]["intent"], "unknown")
        self.assertEqual(outcomes[1]["tag"], "refused")
        self.assertFalse(home.lights["bedroom"])
        self.assertFalse(home.devices["bedroom"])

    def test_pronoun_piece_still_refused(self):
        home, _parsed, outcomes = run_utterance(
            "turn on the kitchen light and switch it off")
        self.assertEqual(outcomes[1]["tag"], "refused")
        self.assertTrue(outcomes[1]["error"])
        self.assertTrue(home.lights["kitchen"])

    def test_multi_limit_unchanged_with_typos(self):
        text = ("turn on the kichen light, turn on the ovem, "
                "turn on the air conditioner, turn on the water heater, "
                "and open the windo")
        _home, parsed, outcomes = run_utterance(text)
        self.assertTrue(parsed["limited"])
        self.assertEqual(len(outcomes), MAX_COMMANDS + 1)
        self.assertEqual(outcomes[-1]["source"], "limit")
        self.assertIn("first %d" % MAX_COMMANDS, outcomes[-1]["message"])

    def test_correct_multi_unchanged(self):
        home, parsed, outcomes = run_utterance(
            "turn off the oven, then lock the door")
        self.assertEqual([o["intent"] for o in outcomes],
                         ["kitchen_device_off", "door_lock"])
        self.assertEqual(parsed["corrections"], [])
        self.assertFalse(home.devices["kitchen"])
        self.assertTrue(home.door_locked)


class TestSTTVariantGroup(unittest.TestCase):
    """The new dataset used by the app accuracy panel."""

    def test_stt_variants_all_correct(self):
        accuracy, details = evaluate_with_rules(MODEL, get_stt_variant_test())
        wrong = []
        for sentence, expected, intent, _confidence in details:
            if expected == "SAFE":
                good = intent == "unknown"
            else:
                good = intent == expected
            if not good:
                wrong.append((sentence, expected, intent))
        self.assertEqual(wrong, [], "mis-scored variants: %s" % wrong)
        self.assertEqual(accuracy, 1.0)


class TestRegression(unittest.TestCase):
    """Every previously supported behavior must keep its score."""

    def test_single_v2_end_to_end(self):
        accuracy, _ = evaluate_with_rules(MODEL, get_single_test_v2())
        self.assertEqual(accuracy, 1.0)

    def test_garbage_still_handled(self):
        accuracy, _ = evaluate_with_rules(MODEL, get_garbage_test())
        self.assertEqual(accuracy, 1.0)

    def test_multi_test(self):
        cmd_acc, sent_acc, rows = score_multi_test(MODEL)
        self.assertEqual(cmd_acc, 1.0, "rows: %s" % rows)
        self.assertEqual(sent_acc, 1.0, "rows: %s" % rows)

    def test_old_accuracy_set(self):
        accuracy, _ = evaluate_accuracy(MODEL)
        self.assertEqual(accuracy, 1.0)

    def test_classifier_alone_unchanged(self):
        accuracy, _ = evaluate_single_v2(MODEL)
        self.assertEqual(accuracy, 0.95)

    def test_ambiguity_singles_all_safe(self):
        for text, expected in get_ambiguity_test():
            self.assertEqual(expected, "SAFE")
            result = parse_command(text, MODEL)
            self.assertEqual(result["intent"], "unknown", text)

    def test_ambiguity_multi_all_safe(self):
        for text, expected in get_ambiguity_multi_test():
            home, _parsed, outcomes = run_utterance(text)
            self.assertEqual(len(outcomes), len(expected))
            for outcome in outcomes:
                self.assertIn(outcome["tag"], ("refused", "not understood"))
                self.assertFalse(outcome["changed_state"])
            self.assertFalse(any(home.lights.values()))
            self.assertFalse(any(home.devices.values()))

    def test_multi_test_rows_exist(self):
        self.assertTrue(get_multi_test())


if __name__ == "__main__":
    unittest.main(verbosity=2)
