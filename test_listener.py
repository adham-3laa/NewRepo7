"""Tests for continuous listening: VAD segmentation and background pipeline.

Run with:
    python -m unittest test_listener -v

No microphone and no WebRTC connection are needed: 30 ms PCM frames are
generated in memory and the voice activity detector is scripted, so every
test is deterministic. The whisper transcription function is injected as
a fake as well; only the real command pipeline (command_parser /
multi_command / SmartHome) is exercised end to end.
"""

import threading
import time
import unittest

import numpy as np

from home import SmartHome
from intent_model import load_or_train
from listener import (
    SAMPLE_RATE,
    VAD_FRAME_BYTES,
    ListenerHandle,
    ListenerWorker,
    PcmConverter,
    ResultDrainer,
    SegmenterConfig,
    UtteranceSegmenter,
)
from multi_command import execute_multi, parse_multi

MODEL = None

SILENCE_FRAME = b"\x00" * VAD_FRAME_BYTES
SPEECH_FRAME = b"\x11" * VAD_FRAME_BYTES
ALT_FRAME = b"\x22" * VAD_FRAME_BYTES  # Frame that is NOT scripted as speech.


def setUpModule():
    """Load the cached intent classifier once for the whole module."""
    global MODEL
    MODEL = load_or_train()


def scripted_vad(frame):
    """Deterministic VAD: speech exactly when the frame starts with 0x11."""
    return frame[:1] == b"\x11"


def frames(count, speech):
    """Repeat one 30 ms VAD frame `count` times."""
    return (SPEECH_FRAME if speech else SILENCE_FRAME) * count


def make_segmenter(**kwargs):
    return UtteranceSegmenter(SegmenterConfig(**kwargs),
                              is_speech=scripted_vad)


def wait_drained(drainer, worker, count, timeout=5.0):
    """Poll like the Streamlit fragment until `count` fresh results exist."""
    deadline = time.monotonic() + timeout
    results = []
    while time.monotonic() < deadline:
        results.extend(drainer.drain(worker))
        if len(results) >= count:
            return results
        time.sleep(0.02)
    return results


def wait_until(predicate, timeout=5.0):
    """Poll a predicate with a deadline (thread hand-off synchronization)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def apply_results(results, home):
    """Mirror of the app thread: run the existing pipeline, once per result.

    Error results and empty transcripts change nothing (the app shows a
    message for empty speech but never executes an action).
    """
    outcomes = []
    for result in results:
        if result.get("error"):
            continue
        transcript = (result.get("transcript") or "").strip()
        if not transcript:
            continue
        parsed = parse_multi(transcript, MODEL)
        outcomes.extend(execute_multi(home, parsed))
    return outcomes


class FakeTranscriber:
    """Injectable transcription function that records its calls."""

    def __init__(self, transcript="", error=None, gate=None):
        self.transcript = transcript
        self.error = error
        self.gate = gate  # Optional Event that blocks a call (in-flight test).
        self.calls = []
        self.started = threading.Event()
        self.wav_existed = False

    def __call__(self, wav_path):
        import os
        self.wav_existed = os.path.exists(wav_path)
        self.calls.append(wav_path)
        self.started.set()
        if self.gate is not None:
            self.gate.wait(10.0)
        if self.error is not None:
            raise RuntimeError(self.error)
        return self.transcript


class SpyWorker:
    """Minimal worker stand-in that records the PCM routed to it."""

    def __init__(self):
        self.received = []

    def push_pcm(self, data):
        self.received.append(data)
        return True


class TestSegmentation(unittest.TestCase):
    """VAD utterance segmentation without any audio hardware."""

    def test_silence_alone_produces_no_utterance(self):
        segmenter = make_segmenter()
        events = segmenter.feed(frames(100, False))  # 3 seconds
        self.assertEqual(events, [])
        self.assertEqual(segmenter.state, "idle")

    def test_speech_then_silence_one_utterance(self):
        segmenter = make_segmenter()
        events = segmenter.feed(frames(10, True) + frames(40, False))
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["kind"], "finalized")
        self.assertFalse(event["truncated"])
        # No silence was fed before the speech, so there is no pre-roll:
        # 10 speech frames + 27 trailing silence frames (the 27th frame
        # reaches the 800 ms timeout).
        self.assertEqual(len(event["pcm"]), (0 + 10 + 27) * VAD_FRAME_BYTES)
        self.assertEqual(event["pcm"].count(SPEECH_FRAME), 10)

    def test_short_pause_does_not_split_utterance(self):
        segmenter = make_segmenter()
        events = segmenter.feed(frames(10, True) + frames(10, False)   # 300 ms
                                + frames(10, True) + frames(30, False))
        self.assertEqual(len(events), 1)
        # Both speech bursts are inside the single utterance.
        self.assertEqual(events[0]["pcm"].count(SPEECH_FRAME), 20)

    def test_preroll_preserves_the_speech_onset(self):
        segmenter = make_segmenter(preroll_ms=210)  # exactly 7 frames
        events = segmenter.feed(ALT_FRAME * 12 + frames(10, True)
                                + frames(40, False))
        self.assertEqual(len(events), 1)
        pcm = events[0]["pcm"]
        # The last 200 ms before speech are kept, in order.
        self.assertEqual(pcm[:7 * VAD_FRAME_BYTES], ALT_FRAME * 7)
        self.assertEqual(pcm[7 * VAD_FRAME_BYTES:8 * VAD_FRAME_BYTES],
                         SPEECH_FRAME)

    def test_preroll_zero_starts_at_speech(self):
        segmenter = make_segmenter(preroll_ms=0)
        events = segmenter.feed(ALT_FRAME * 5 + frames(10, True)
                                + frames(30, False))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["pcm"][:VAD_FRAME_BYTES], SPEECH_FRAME)

    def test_too_short_noise_segment_is_discarded(self):
        segmenter = make_segmenter()
        events = segmenter.feed(frames(5, True) + frames(30, False))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["kind"], "discarded")
        self.assertEqual(events[0]["reason"], "speech too short")
        self.assertEqual(segmenter.state, "idle")

    def test_maximum_duration_rejected_and_suppressed(self):
        segmenter = make_segmenter(max_utterance_ms=1000)
        events = segmenter.feed(frames(60, True))  # 1.8 s continuous speech
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["kind"], "discarded")
        self.assertEqual(events[0]["reason"], "maximum duration reached")
        # The rest of this speech must not become a second utterance.
        self.assertEqual(segmenter.feed(frames(20, True)), [])
        self.assertEqual(segmenter.state, "suppressed")
        # Suppression ends only at a reliable silence boundary.
        self.assertEqual(segmenter.feed(frames(25, False)), [])
        self.assertEqual(segmenter.state, "suppressed")
        self.assertEqual(segmenter.feed(frames(5, False)), [])
        self.assertEqual(segmenter.state, "idle")
        # Listening recovers completely after the boundary.
        events = segmenter.feed(frames(10, True) + frames(30, False))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["kind"], "finalized")

    def test_truncated_utterance_emitted_only_when_allowed(self):
        segmenter = make_segmenter(max_utterance_ms=1000,
                                   emit_truncated=True)
        events = segmenter.feed(frames(40, True))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["kind"], "finalized")
        self.assertTrue(events[0]["truncated"])
        self.assertEqual(segmenter.state, "suppressed")

    def test_abort_suppresses_until_silence_boundary(self):
        segmenter = make_segmenter()
        self.assertEqual(segmenter.feed(frames(5, True)), [])
        event = segmenter.abort(suppress=True)  # e.g. audio overflow
        self.assertEqual(event["kind"], "discarded")
        self.assertEqual(segmenter.state, "suppressed")
        self.assertEqual(segmenter.feed(frames(10, True)), [])
        self.assertEqual(segmenter.feed(frames(30, False)), [])
        self.assertEqual(segmenter.state, "idle")
        events = segmenter.feed(frames(10, True) + frames(30, False))
        self.assertEqual(len(events), 1)

    def test_abort_without_suppression_resumes_immediately(self):
        segmenter = make_segmenter()
        segmenter.feed(frames(5, True))
        segmenter.abort(suppress=False)  # explicit house reset
        self.assertEqual(segmenter.state, "idle")
        events = segmenter.feed(frames(10, True) + frames(30, False))
        self.assertEqual(len(events), 1)

    def test_arbitrary_chunk_sizes_produce_identical_audio(self):
        whole = frames(10, True) + frames(40, False)
        segmenter_a = make_segmenter()
        events_a = segmenter_a.feed(whole)
        segmenter_b = make_segmenter()
        events_b = []
        for start in range(0, len(whole), 997):  # odd, non-frame-aligned
            events_b.extend(segmenter_b.feed(whole[start:start + 997]))
        self.assertEqual(len(events_a), 1)
        self.assertEqual(len(events_b), 1)
        self.assertEqual(events_a[0]["pcm"], events_b[0]["pcm"])

    def test_leftover_samples_are_kept_for_the_next_frame(self):
        segmenter = make_segmenter()
        # Half a frame produces no VAD frame yet.
        self.assertEqual(segmenter.feed(SPEECH_FRAME[:500]), [])
        self.assertEqual(segmenter.state, "idle")
        # The remaining 460 bytes complete exactly one 30 ms frame and
        # open an utterance.
        self.assertEqual(segmenter.feed(SPEECH_FRAME[500:]), [])
        self.assertEqual(segmenter.state, "speech")

    def test_default_webrtcvad_path_handles_silence(self):
        segmenter = UtteranceSegmenter(SegmenterConfig())
        self.assertEqual(segmenter.feed(SILENCE_FRAME * 34), [])
        self.assertEqual(segmenter.state, "idle")


class TestPcmConverter(unittest.TestCase):
    """PyAV conversion with resampler state and arbitrary output sizes."""

    def test_stereo_48k_becomes_mono_16k(self):
        import av
        converter = PcmConverter()
        total = 0
        for _ in range(5):
            # Packed s16 stereo expects shape (1, samples * channels).
            samples = np.zeros((1, 4800 * 2), dtype=np.int16)  # 0.1 s 48k
            frame = av.AudioFrame.from_ndarray(samples, format="s16",
                                               layout="stereo")
            frame.sample_rate = 48000
            for chunk in converter.convert(frame):
                self.assertEqual(len(chunk) % 2, 0)  # whole s16 samples
                total += len(chunk)
        expected = int(0.5 * SAMPLE_RATE) * 2
        # The resampler may buffer a little; one frame of tolerance.
        self.assertLessEqual(abs(total - expected), 2 * VAD_FRAME_BYTES)

    def test_resampler_state_survives_many_small_frames(self):
        import av
        converter = PcmConverter()
        total = 0
        outputs = 0
        for _ in range(50):
            samples = np.zeros((1, 512), dtype=np.int16)  # 48 kHz mono
            frame = av.AudioFrame.from_ndarray(samples, format="s16",
                                               layout="mono")
            frame.sample_rate = 48000
            chunks = converter.convert(frame)
            outputs += len(chunks)
            total += sum(len(chunk) for chunk in chunks)
        expected = int(50 * 512 / 48000.0 * SAMPLE_RATE) * 2
        self.assertGreater(outputs, 0)
        self.assertLessEqual(abs(total - expected), 2 * VAD_FRAME_BYTES)


class TestListenerHandle(unittest.TestCase):
    """Stable callback-to-worker handoff and buffering."""

    def test_audio_before_worker_is_flushed_not_lost(self):
        handle = ListenerHandle()
        handle.push_pcm(SPEECH_FRAME)
        handle.push_pcm(SILENCE_FRAME)
        spy = SpyWorker()
        handle.set_worker(spy)
        self.assertEqual(spy.received, [SPEECH_FRAME, SILENCE_FRAME])
        handle.push_pcm(SPEECH_FRAME)
        self.assertEqual(spy.received,
                         [SPEECH_FRAME, SILENCE_FRAME, SPEECH_FRAME])

    def test_buffer_is_bounded(self):
        handle = ListenerHandle()
        handle.push_pcm(b"\x00" * (ListenerHandle.MAX_BUFFER_BYTES + 1000))
        spy = SpyWorker()
        handle.set_worker(spy)
        total = sum(len(chunk) for chunk in spy.received)
        self.assertLessEqual(total, ListenerHandle.MAX_BUFFER_BYTES)

    def test_stop_discards_incoming_audio_until_restart(self):
        handle = ListenerHandle()
        spy = SpyWorker()
        handle.set_worker(spy)
        handle.push_pcm(SPEECH_FRAME)
        old = handle.stop()
        self.assertIs(old, spy)
        handle.push_pcm(SILENCE_FRAME)  # in flight after STOP: dropped
        self.assertEqual(len(spy.received), 1)
        spy2 = SpyWorker()
        handle.set_worker(spy2)  # nothing from before the stop is flushed
        self.assertEqual(spy2.received, [])
        handle.push_pcm(SPEECH_FRAME)
        self.assertEqual(spy2.received, [SPEECH_FRAME])

    def test_swap_never_routes_to_the_old_worker(self):
        handle = ListenerHandle()
        spy1, spy2 = SpyWorker(), SpyWorker()
        handle.set_worker(spy1)
        handle.push_pcm(SPEECH_FRAME)
        old = handle.swap_worker(spy2)
        self.assertIs(old, spy1)
        handle.push_pcm(SILENCE_FRAME)
        self.assertEqual(len(spy1.received), 1)  # old worker got nothing new
        self.assertEqual(len(spy2.received), 1)

    def test_audio_ended_flag_resets_on_attach(self):
        handle = ListenerHandle()
        self.assertFalse(handle.audio_ended())
        handle.mark_audio_ended()
        self.assertTrue(handle.audio_ended())
        handle.set_worker(SpyWorker())
        self.assertFalse(handle.audio_ended())


class TestWorkerPipeline(unittest.TestCase):
    """Worker threads, generations and exactly-once result draining."""

    def make_worker(self, fake, **kwargs):
        return ListenerWorker(fake, vad_fn=scripted_vad, **kwargs)

    def test_empty_transcription_causes_no_action(self):
        fake = FakeTranscriber(transcript="")
        worker = self.make_worker(fake)
        try:
            worker.push_pcm(frames(10, True) + frames(40, False))
            drainer = ResultDrainer()
            results = wait_drained(drainer, worker, 1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["transcript"], "")
            self.assertIsNone(results[0]["error"])
            self.assertTrue(fake.wav_existed)
            home = SmartHome()
            self.assertEqual(apply_results(results, home), [])
            self.assertFalse(any(home.lights.values()))
            self.assertFalse(any(home.devices.values()))
        finally:
            worker.stop()

    def test_failed_transcription_causes_no_action(self):
        fake = FakeTranscriber(error="model exploded")
        worker = self.make_worker(fake)
        try:
            worker.push_pcm(frames(10, True) + frames(40, False))
            drainer = ResultDrainer()
            results = wait_drained(drainer, worker, 1)
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0]["error"])
            home = SmartHome()
            self.assertEqual(apply_results(results, home), [])
        finally:
            worker.stop()

    def test_open_window_command_executes_exactly_once(self):
        fake = FakeTranscriber(transcript="open the window")
        worker = self.make_worker(fake)
        try:
            worker.push_pcm(frames(10, True) + frames(40, False))
            drainer = ResultDrainer()
            results = wait_drained(drainer, worker, 1)
            self.assertEqual(len(results), 1)
            home = SmartHome()
            outcomes = apply_results(results, home)
            self.assertTrue(home.window_open)
            self.assertEqual(len([o for o in outcomes
                                  if o["tag"] == "success"]), 1)
            # Repeated polling must not deliver or execute it again.
            self.assertEqual(drainer.drain(worker), [])
            self.assertEqual(apply_results(drainer.drain(worker), home), [])
            self.assertTrue(home.window_open)
        finally:
            worker.stop()

    def test_second_utterance_without_restarting(self):
        fake = FakeTranscriber(transcript="turn on the kitchen light")
        worker = self.make_worker(fake)
        try:
            drainer = ResultDrainer()
            worker.push_pcm(frames(10, True) + frames(40, False))
            first = wait_drained(drainer, worker, 1)
            worker.push_pcm(frames(10, True) + frames(40, False))
            second = wait_drained(drainer, worker, 1)
            self.assertEqual(len(first), 1)
            self.assertEqual(len(second), 1)
            self.assertNotEqual(first[0]["uid"], second[0]["uid"])
            home = SmartHome()
            outcomes = apply_results(first + second, home)
            self.assertEqual(len(outcomes), 2)
            self.assertTrue(home.lights["kitchen"])
            self.assertEqual(worker.status()["processed"], 2)
        finally:
            worker.stop()

    def test_unknown_phrase_changes_no_device_state(self):
        fake = FakeTranscriber(transcript="make coffee")
        worker = self.make_worker(fake)
        try:
            worker.push_pcm(frames(10, True) + frames(40, False))
            drainer = ResultDrainer()
            results = wait_drained(drainer, worker, 1)
            home = SmartHome()
            outcomes = apply_results(results, home)
            self.assertTrue(outcomes)
            for outcome in outcomes:
                self.assertIn(outcome["tag"], ("refused", "not understood"))
                self.assertFalse(outcome["changed_state"])
            self.assertFalse(any(home.lights.values()))
            self.assertFalse(any(home.devices.values()))
            self.assertFalse(home.door_open)
            self.assertFalse(home.window_open)
        finally:
            worker.stop()

    def test_status_query_still_works_and_changes_nothing(self):
        fake = FakeTranscriber(transcript="what is the status of the house")
        worker = self.make_worker(fake)
        try:
            worker.push_pcm(frames(10, True) + frames(40, False))
            drainer = ResultDrainer()
            results = wait_drained(drainer, worker, 1)
            home = SmartHome()
            outcomes = apply_results(results, home)
            self.assertEqual([o["intent"] for o in outcomes], ["status"])
            self.assertFalse(any(home.lights.values()))
        finally:
            worker.stop()

    def test_stop_prevents_further_processing(self):
        fake = FakeTranscriber(transcript="open the window")
        worker = self.make_worker(fake)
        worker.push_pcm(frames(10, True) + frames(40, False))
        drainer = ResultDrainer()
        wait_drained(drainer, worker, 1)  # first utterance processed
        worker.stop()
        self.assertFalse(worker.push_pcm(frames(10, True) + frames(40, False)))
        time.sleep(0.1)
        self.assertEqual(drainer.drain(worker), [])
        self.assertEqual(worker.drain_results(), [])
        self.assertFalse(worker.status()["running"])

    def test_stop_discards_undrained_results(self):
        fake = FakeTranscriber(transcript="open the window")
        worker = self.make_worker(fake)
        worker.push_pcm(frames(10, True) + frames(40, False))
        self.assertTrue(wait_until(
            lambda: worker.status()["processed"] >= 1))
        worker.stop()  # result was published but never drained
        self.assertEqual(worker.drain_results(), [])
        self.assertEqual(ResultDrainer().drain(worker), [])

    def test_reset_discards_in_flight_transcription(self):
        gate = threading.Event()
        fake = FakeTranscriber(transcript="open the window", gate=gate)
        worker = self.make_worker(fake)
        try:
            worker.push_pcm(frames(10, True) + frames(40, False))
            self.assertTrue(fake.started.wait(5.0))
            worker.reset()  # House reset while Whisper is running.
            gate.set()
            self.assertTrue(wait_until(
                lambda: worker.status()["pending"] < 1))
            time.sleep(0.2)
            self.assertEqual(worker.drain_results(), [])
            self.assertEqual(ResultDrainer().drain(worker), [])
        finally:
            gate.set()
            worker.stop()

    def test_reset_invalidates_pending_and_keeps_listening(self):
        fake = FakeTranscriber(transcript="open the window")
        worker = self.make_worker(fake)
        try:
            worker.push_pcm(frames(10, True) + frames(40, False))
            self.assertTrue(wait_until(
                lambda: worker.status()["processed"] >= 1))
            worker.reset()  # Undrained pre-reset result must be dropped.
            self.assertEqual(worker.drain_results(), [])
            self.assertEqual(worker.generation(), 1)
            # Listening continues; new audio is processed under the new
            # generation and executes normally.
            worker.push_pcm(frames(10, True) + frames(40, False))
            drainer = ResultDrainer()
            results = wait_drained(drainer, worker, 1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["generation"], 1)
            home = SmartHome()
            apply_results(results, home)
            self.assertTrue(home.window_open)
        finally:
            worker.stop()

    def test_overrun_discards_compromised_audio_until_silence(self):
        fake = FakeTranscriber(transcript="open the window")
        # auto_start=False keeps the queue order fully deterministic.
        worker = self.make_worker(fake, auto_start=False, intake_maxsize=1)
        self.assertTrue(worker.push_pcm(SPEECH_FRAME))    # occupies the slot
        self.assertFalse(worker.push_pcm(SPEECH_FRAME))   # overflow
        self.assertFalse(worker.push_pcm(SPEECH_FRAME))   # overflow
        worker.start()
        try:
            # The overrun aborts the compromised speech and suppresses
            # the segmenter until a silence boundary.
            self.assertTrue(wait_until(
                lambda: worker.status()["segment_state"] == "suppressed"))
            worker.push_pcm(frames(30, False))
            self.assertTrue(wait_until(
                lambda: worker.status()["segment_state"] == "idle"))
            # After the boundary, listening recovers and the next
            # utterance is transcribed and executed exactly once.
            worker.push_pcm(frames(10, True) + frames(40, False))
            drainer = ResultDrainer()
            results = wait_drained(drainer, worker, 1)
            self.assertEqual(len(results), 1)
            time.sleep(0.2)
            self.assertEqual(drainer.drain(worker), [])
            self.assertEqual(worker.status()["overruns"], 2)
            home = SmartHome()
            apply_results(results, home)
            self.assertTrue(home.window_open)
        finally:
            worker.stop()

    def test_worker_refuses_audio_after_stop(self):
        fake = FakeTranscriber(transcript="")
        worker = self.make_worker(fake)
        self.assertTrue(worker.push_pcm(SPEECH_FRAME))
        worker.stop()
        self.assertFalse(worker.push_pcm(SPEECH_FRAME))

    def test_status_snapshot_fields(self):
        fake = FakeTranscriber(transcript="")
        worker = ListenerWorker(fake, engine_label="test-engine",
                                vad_fn=scripted_vad)
        try:
            status = worker.status()
            for key in ("generation", "running", "frames", "last_frame_at",
                        "created_at", "segment_state", "pending", "processed",
                        "discarded", "overruns", "last_note", "engine"):
                self.assertIn(key, status)
            self.assertTrue(status["running"])
            self.assertEqual(status["engine"], "test-engine")
            self.assertEqual(status["frames"], 0)
            worker.push_pcm(SILENCE_FRAME)
            self.assertTrue(wait_until(lambda: worker.status()["frames"] == 1))
        finally:
            worker.stop()
        self.assertFalse(worker.status()["running"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
