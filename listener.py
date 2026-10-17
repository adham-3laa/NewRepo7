"""Continuous microphone listening: VAD segmentation + transcription.

The browser streams microphone audio through streamlit-webrtc. Each
incoming PyAV audio frame is converted to 16 kHz mono signed 16-bit
PCM, cut into exact 30 ms frames and classified by a streaming voice
activity detector (webrtcvad). Speech is buffered (including a
pre-roll) until a silence timeout, then a background thread
transcribes the finished utterance with the existing speech.py
pipeline. Only the Streamlit application thread parses, executes and
renders commands.

Threading model (all queues are bounded):
- WebRTC audio callback thread: convert frames, hand PCM to
  ListenerHandle (short, never runs Whisper).
- ListenerHandle: stable thread-safe reference to the current worker,
  shared with the audio callback for the whole Streamlit session.
- Segmentation thread: VAD, utterance state machine, discarding.
- Transcription thread: write a temporary WAV, run the injected
  transcribe function, publish one result per utterance.
- Streamlit app thread: ResultDrainer.drain() -> existing command
  pipeline (command_parser / multi_command / SmartHome).

Safety rules implemented here:
- A maximum-duration utterance is truncated and rejected by default;
  the rest of that speech is ignored until a silence boundary.
- After an audio-queue overflow the compromised utterance is dropped
  and speech is ignored until a silence boundary.
- Every listening session has a generation token. reset() bumps it so
  audio, jobs and results captured before a house reset can never be
  executed afterwards. stop() invalidates everything.
- Every utterance has a uuid; ResultDrainer hands each one out at
  most once for the life of the Streamlit session.
"""

import collections
import os
import queue
import tempfile
import threading
import time
import uuid
import wave

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # bytes per sample (signed 16-bit)
CHANNELS = 1
VAD_FRAME_MS = 30
VAD_FRAME_BYTES = SAMPLE_RATE * VAD_FRAME_MS // 1000 * SAMPLE_WIDTH  # 960

# Stop waiting for a thread to wind down. A running transcription may
# still be in flight; its result is discarded by the generation check.
_JOIN_TIMEOUT = 1.0


def _clear_queue(target):
    """Empty a queue without blocking (thread-safe)."""
    while True:
        try:
            target.get_nowait()
        except queue.Empty:
            return


def _make_vad(mode):
    """Build a webrtcvad predicate for 16 kHz 30 ms frames."""
    try:
        import webrtcvad
    except ImportError as exc:
        raise RuntimeError(
            "Continuous listening needs the 'webrtcvad-wheels' package. "
            "Install the project requirements (pip install -r "
            "requirements.txt) and restart. Details: %s" % exc)
    vad = webrtcvad.Vad(mode)

    def is_speech(frame_bytes):
        return vad.is_speech(frame_bytes, SAMPLE_RATE)

    return is_speech


class SegmenterConfig:
    """Tunable utterance segmentation settings.

    All values are milliseconds except vad_mode (0-3, higher is more
    aggressive). preroll_frames is derived: the number of 30 ms VAD
    frames kept before speech onset so the first phoneme survives.
    """

    def __init__(self, silence_ms=800, min_speech_ms=250, preroll_ms=200,
                 max_utterance_ms=15000, vad_mode=2, emit_truncated=False):
        self.silence_ms = max(VAD_FRAME_MS, int(silence_ms))
        self.min_speech_ms = max(0, int(min_speech_ms))
        self.preroll_ms = max(0, int(preroll_ms))
        self.max_utterance_ms = max(VAD_FRAME_MS, int(max_utterance_ms))
        self.vad_mode = min(3, max(0, int(vad_mode)))
        # When False (default) a maximum-duration utterance is rejected
        # instead of being transcribed as a possibly incomplete command.
        self.emit_truncated = bool(emit_truncated)
        self.preroll_frames = int(round(self.preroll_ms / float(VAD_FRAME_MS)))

    def snapshot(self):
        """Hashable tuple of the tunable values (change detection)."""
        return (self.silence_ms, self.min_speech_ms, self.preroll_ms,
                self.max_utterance_ms, self.vad_mode, self.emit_truncated)


class UtteranceSegmenter:
    """Streaming VAD utterance segmenter.

    feed() accepts PCM bytes in any chunk size. Audio is cut into exact
    30 ms frames and leftover bytes are kept for the next call (PyAV or
    the resampler never guarantee 30 ms alignment). is_speech is
    injected so tests can script speech and silence deterministically;
    the default is a lazily created webrtcvad instance.

    feed()/abort() return event dicts:
    - {"kind": "finalized", "pcm", "duration_ms", "truncated"}
    - {"kind": "discarded", "reason", "duration_ms"}
    """

    def __init__(self, config=None, is_speech=None):
        self.config = config or SegmenterConfig()
        self._is_speech = is_speech
        self._pending = bytearray()
        self._ring = collections.deque(
            maxlen=self.config.preroll_frames)
        self._buffer = []
        self._speech_ms = 0
        self._speech_end_ms = 0
        self._silence_ms = 0
        self._suppressed_ms = 0
        # idle: ignoring silence; speech: buffering an utterance;
        # suppressed: ignoring speech until a silence boundary.
        self._state = "idle"

    @property
    def state(self):
        """One of 'idle', 'speech', 'suppressed'."""
        return self._state

    def feed(self, data):
        """Add PCM bytes. Returns a list of events (usually empty)."""
        events = []
        if data:
            self._pending.extend(data)
        while len(self._pending) >= VAD_FRAME_BYTES:
            frame = bytes(self._pending[:VAD_FRAME_BYTES])
            del self._pending[:VAD_FRAME_BYTES]
            event = self._process_frame(frame)
            if event is not None:
                events.append(event)
        return events

    def abort(self, suppress=True):
        """Drop the partial utterance after an error or reset.

        suppress=True (audio overflow): ignore further speech until a
        silence boundary so corrupted fragments are never executed.
        suppress=False (explicit house reset): resume listening at
        once; the caller must already have invalidated the old audio.
        Returns a discarded event when an utterance was in progress.
        """
        had_audio = bool(self._buffer)
        duration_ms = len(self._buffer) * VAD_FRAME_MS
        self._buffer = []
        self._speech_ms = 0
        self._speech_end_ms = 0
        self._silence_ms = 0
        self._ring.clear()
        if suppress:
            self._state = "suppressed"
            self._suppressed_ms = 0
        else:
            self._state = "idle"
        if had_audio:
            return {"kind": "discarded", "reason": "audio interrupted",
                    "duration_ms": duration_ms}
        return None

    def _classify(self, frame):
        """True when the 30 ms frame contains speech."""
        if self._is_speech is None:
            self._is_speech = _make_vad(self.config.vad_mode)
        try:
            return bool(self._is_speech(frame))
        except Exception:
            return False  # A bad frame must never kill the listener.

    def _process_frame(self, frame):
        speech = self._classify(frame)
        if self._state == "suppressed":
            return self._process_suppressed(speech)
        if self._state == "idle":
            if not speech:
                if self.config.preroll_frames:
                    self._ring.append(frame)
                return None
            # Speech onset: keep the pre-roll so the first phoneme
            # (and a little room tone) is not cut off.
            self._buffer = list(self._ring)
            self._ring.clear()
            self._buffer.append(frame)
            self._speech_ms = VAD_FRAME_MS
            self._speech_end_ms = len(self._buffer) * VAD_FRAME_MS
            self._silence_ms = 0
            self._state = "speech"
            return None
        # state == "speech": keep appending, including pauses shorter
        # than the silence timeout so a sentence is not split early.
        self._buffer.append(frame)
        if speech:
            self._speech_ms += VAD_FRAME_MS
            self._speech_end_ms = len(self._buffer) * VAD_FRAME_MS
            self._silence_ms = 0
        else:
            self._silence_ms += VAD_FRAME_MS
        # The maximum applies to the continuous speech itself, not to
        # the trailing silence that is kept to confirm the end.
        if self._speech_end_ms >= self.config.max_utterance_ms:
            return self._finalize(truncated=True)
        if self._silence_ms >= self.config.silence_ms:
            return self._finalize(truncated=False)
        return None

    def _process_suppressed(self, speech):
        """Ignore speech until a reliable silence boundary is seen."""
        if speech:
            self._suppressed_ms = 0
            return None
        self._suppressed_ms += VAD_FRAME_MS
        if self._suppressed_ms >= self.config.silence_ms:
            self._state = "idle"
            self._suppressed_ms = 0
            self._ring.clear()
        return None

    def _finalize(self, truncated):
        """Close the current utterance and return an event."""
        pcm = b"".join(self._buffer)
        duration_ms = len(self._buffer) * VAD_FRAME_MS
        speech_ms = self._speech_ms
        self._buffer = []
        self._speech_ms = 0
        self._speech_end_ms = 0
        self._silence_ms = 0
        self._ring.clear()
        if truncated:
            # Maximum duration: the utterance may be incomplete, so
            # ignore the rest of this speech until a silence boundary.
            self._state = "suppressed"
            self._suppressed_ms = 0
            if self.config.emit_truncated:
                return {"kind": "finalized", "pcm": pcm,
                        "duration_ms": duration_ms, "truncated": True}
            return {"kind": "discarded", "reason": "maximum duration reached",
                    "duration_ms": duration_ms}
        self._state = "idle"
        if speech_ms < self.config.min_speech_ms:
            return {"kind": "discarded", "reason": "speech too short",
                    "duration_ms": speech_ms}
        return {"kind": "finalized", "pcm": pcm, "duration_ms": duration_ms,
                "truncated": False}


class PcmConverter:
    """Convert PyAV audio frames to 16 kHz mono signed 16-bit PCM.

    One instance keeps the resampler state, so continuity is preserved
    across input frames. convert() returns EVERY chunk the resampler
    produces (it may be empty, more than one frame, or not a multiple
    of 30 ms); the segmenter handles arbitrary chunk sizes and keeps
    leftover samples itself.
    """

    def __init__(self):
        import av
        self._resampler = av.AudioResampler(format="s16", layout="mono",
                                            rate=SAMPLE_RATE)

    def convert(self, frame):
        """Return a list of PCM byte strings for one input frame."""
        outputs = self._resampler.resample(frame) or []
        chunks = []
        for resampled in outputs:
            chunks.append(resampled.to_ndarray().tobytes())
        return chunks


class ListenerHandle:
    """Stable, thread-safe bridge between the audio callback and workers.

    The WebRTC audio callback holds a reference to this object for the
    whole Streamlit session, so replacing the worker (start, stop,
    settings change, reset) can never route frames to a stale worker.
    While no worker is attached yet, a bounded amount of converted PCM
    is kept so the beginning of the first utterance is not lost in the
    short window between the stream starting and the worker being
    created. After an explicit stop, incoming frames are discarded
    until a new worker is attached.
    """

    MAX_BUFFER_BYTES = SAMPLE_RATE * SAMPLE_WIDTH  # one second

    def __init__(self):
        self._lock = threading.Lock()
        self._worker = None
        self._buffer = collections.deque()
        self._buffer_bytes = 0
        self._discard = False
        self._converter = None
        self._converter_error = None
        self._audio_ended = False

    def push_frame(self, frame):
        """Audio-callback entry point: convert and route one PyAV frame."""
        with self._lock:
            if self._converter is None and self._converter_error is None:
                try:
                    self._converter = PcmConverter()
                except Exception as exc:
                    self._converter_error = str(exc)
            converter = self._converter
        if converter is None:
            return False
        try:
            chunks = converter.convert(frame)
        except Exception:
            return False  # Malformed frame: drop it, keep the stream.
        return self.push_chunks(chunks)

    def push_pcm(self, data):
        """Route already-converted PCM (used by tests and by push_frame)."""
        return self.push_chunks([data])

    def push_chunks(self, chunks):
        """Route a list of PCM chunks to the current worker or buffer."""
        routed = False
        with self._lock:
            if self._discard:
                return False
            for chunk in chunks:
                if not chunk:
                    continue
                worker = self._worker
                if worker is not None:
                    worker.push_pcm(chunk)
                    routed = True
                else:
                    self._buffer.append(chunk)
                    self._buffer_bytes += len(chunk)
                    while (self._buffer_bytes > self.MAX_BUFFER_BYTES
                           and self._buffer):
                        self._buffer_bytes -= len(self._buffer.popleft())
        return routed

    def set_worker(self, worker):
        """Attach a worker (listening start). Flushes buffered audio."""
        self._attach(worker)

    def swap_worker(self, worker):
        """Replace the worker, returning the previous one (may be None)."""
        with self._lock:
            old = self._worker
        self._attach(worker)
        return old

    def stop(self):
        """Detach the worker; discard buffers and further incoming audio."""
        with self._lock:
            old = self._worker
            self._worker = None
            self._discard = True
            self._audio_ended = False
            self._buffer.clear()
            self._buffer_bytes = 0
        return old

    def mark_audio_ended(self):
        """Called from on_audio_ended (aiortc thread). Never raises."""
        with self._lock:
            self._audio_ended = True

    def clear_audio_ended(self):
        """Forget an end event once frames provably flow again."""
        with self._lock:
            self._audio_ended = False

    def audio_ended(self):
        """True when the input media track ended unexpectedly."""
        with self._lock:
            return self._audio_ended

    def conversion_error(self):
        """Message when PyAV conversion could not even be initialized."""
        with self._lock:
            return self._converter_error

    def _attach(self, worker):
        """Point at a new worker and flush the pre-worker buffer to it."""
        with self._lock:
            self._worker = worker
            self._discard = False
            self._audio_ended = False
            buffered = list(self._buffer)
            self._buffer.clear()
            self._buffer_bytes = 0
        for chunk in buffered:
            worker.push_pcm(chunk)


class ListenerWorker:
    """Background segmentation and transcription for one listening session.

    push_pcm() is called from the audio callback thread and only
    enqueues (bounded). stop() is final: a stopped worker refuses new
    audio and never publishes results again. reset() bumps the
    generation so audio, jobs and results captured before a house
    reset can never execute afterwards.
    """

    INTAKE_MAXSIZE = 200    # Bounded raw PCM chunks.
    JOB_MAXSIZE = 8         # Bounded finalized utterances awaiting Whisper.

    def __init__(self, transcribe_fn, engine_label="local", config=None,
                 vad_fn=None, auto_start=True, intake_maxsize=None,
                 job_maxsize=None):
        self._transcribe_fn = transcribe_fn
        self._engine_label = engine_label
        self._config = config or SegmenterConfig()
        self._segmenter = UtteranceSegmenter(self._config, is_speech=vad_fn)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._running = True
        self._generation = 0
        self._reset_requested = False
        self._overrun_requested = False
        self._frames = 0
        self._last_frame_at = None
        self._segment_state = "idle"
        self._pending_jobs = 0
        self._transcribing = False
        self._processed = 0
        self._discarded = 0
        self._overruns = 0
        self._last_note = None
        self._created_at = time.monotonic()
        self._intake = queue.Queue(
            maxsize=intake_maxsize if intake_maxsize is not None
            else self.INTAKE_MAXSIZE)
        self._jobs = queue.Queue(
            maxsize=job_maxsize if job_maxsize is not None
            else self.JOB_MAXSIZE)
        self._results = queue.Queue()
        self._seg_thread = None
        self._trans_thread = None
        if auto_start:
            self.start()

    # ---- lifecycle ---------------------------------------------------
    def start(self):
        """Start the segmentation and transcription threads once."""
        with self._lock:
            if not self._running or self._seg_thread is not None:
                return
            self._seg_thread = threading.Thread(
                target=self._segment_loop, name="listener-segment",
                daemon=True)
            self._trans_thread = threading.Thread(
                target=self._transcribe_loop, name="listener-transcribe",
                daemon=True)
        self._seg_thread.start()
        self._trans_thread.start()

    def stop(self, join_timeout=_JOIN_TIMEOUT):
        """Final stop: invalidate pending audio, jobs and results."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._stop_event.set()
        _clear_queue(self._intake)
        _clear_queue(self._jobs)
        _clear_queue(self._results)
        for thread in (self._seg_thread, self._trans_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=join_timeout)

    def reset(self):
        """Invalidate everything captured before a house reset.

        Bumps the generation, aborts the partial utterance (without
        suppression) and clears all queues. Chunks carry the
        generation they were captured under, so pre-reset audio is
        dropped even if it is still in flight.
        """
        with self._lock:
            if not self._running:
                return
            self._generation += 1
            self._reset_requested = True
        _clear_queue(self._intake)
        _clear_queue(self._jobs)
        _clear_queue(self._results)

    def generation(self):
        """Current generation token (bumped by reset())."""
        with self._lock:
            return self._generation

    def is_running(self):
        with self._lock:
            return self._running

    # ---- audio intake (audio callback thread) ------------------------
    def push_pcm(self, data):
        """Hand converted PCM to the segmentation thread. Never blocks."""
        if not data:
            return False
        with self._lock:
            if not self._running:
                return False
            generation = self._generation
            self._frames += 1
            self._last_frame_at = time.monotonic()
        try:
            self._intake.put_nowait((generation, data))
            return True
        except queue.Full:
            # Overflow: the audio stream is compromised, so drop the
            # frame and let the segmenter discard current speech until
            # a silence boundary (handled in the segment loop).
            with self._lock:
                if self._running:
                    self._overruns += 1
                    self._overrun_requested = True
                    self._last_note = ("Audio buffer overflow; discarding "
                                       "the current speech until the next "
                                       "pause.")
            return False

    # ---- results (application thread) --------------------------------
    def drain_results(self):
        """Pop every ready result. Called only from the app thread."""
        results = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                return results

    def status(self):
        """Thread-safe snapshot for the Streamlit status area."""
        with self._lock:
            return {
                "generation": self._generation,
                "running": self._running,
                "frames": self._frames,
                "last_frame_at": self._last_frame_at,
                "created_at": self._created_at,
                "segment_state": self._segment_state,
                "pending": self._pending_jobs + (1 if self._transcribing
                                                 else 0),
                "processed": self._processed,
                "discarded": self._discarded,
                "overruns": self._overruns,
                "last_note": self._last_note,
                "engine": self._engine_label,
            }

    # ---- segmentation thread -----------------------------------------
    def _segment_loop(self):
        while not self._stop_event.is_set():
            if self._take_flag("_reset_requested"):
                self._record_event(self._segmenter.abort(suppress=False))
            if self._take_flag("_overrun_requested"):
                self._record_event(self._segmenter.abort(suppress=True))
            try:
                generation, data = self._intake.get(timeout=0.1)
            except queue.Empty:
                continue
            if self._stop_event.is_set():
                return
            if generation != self.generation():
                continue  # Captured before a reset: never segment it.
            for event in self._segmenter.feed(data):
                if self._stop_event.is_set():
                    return
                self._handle_segment_event(event)
            with self._lock:
                self._segment_state = self._segmenter.state

    def _take_flag(self, name):
        """Consume a one-shot request flag set by another thread."""
        with self._lock:
            if getattr(self, name):
                setattr(self, name, False)
                return True
            return False

    def _record_event(self, event):
        """Record a segmenter event that produced no utterance."""
        if event is None:
            return
        with self._lock:
            self._discarded += 1
            self._last_note = "Discarded: %s." % event.get("reason", "audio")

    def _handle_segment_event(self, event):
        """Queue a finalized utterance or record why it was dropped."""
        if event is None:
            return
        if event["kind"] == "discarded":
            self._record_event(event)
            return
        if event["kind"] != "finalized":
            return
        if event.get("truncated") and not self._config.emit_truncated:
            # Defense in depth: a truncated utterance is never executed
            # unless truncation was explicitly allowed.
            with self._lock:
                self._discarded += 1
                self._last_note = ("Long speech discarded; please speak a "
                                   "shorter command.")
            return
        job = {
            "uid": uuid.uuid4().hex,
            "generation": self.generation(),
            "pcm": event["pcm"],
            "duration_ms": event["duration_ms"],
            "truncated": event.get("truncated", False),
        }
        try:
            self._jobs.put_nowait(job)
            with self._lock:
                self._pending_jobs += 1
        except queue.Full:
            with self._lock:
                self._discarded += 1
                self._last_note = ("Transcription queue full; utterance "
                                   "discarded.")

    # ---- transcription thread ----------------------------------------
    def _transcribe_loop(self):
        while not self._stop_event.is_set():
            try:
                job = self._jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            with self._lock:
                self._pending_jobs = max(0, self._pending_jobs - 1)
                if not self._running or job["generation"] != self._generation:
                    continue
                self._transcribing = True
            transcript = ""
            error = None
            try:
                transcript = (self._run_transcribe(job["pcm"]) or "").strip()
            except Exception as exc:
                error = "Transcription failed: %s" % exc
            finally:
                with self._lock:
                    self._transcribing = False
            with self._lock:
                stale = (not self._running
                         or job["generation"] != self._generation)
                if not stale:
                    self._processed += 1
            if stale:
                # A reset or stop happened while Whisper was running:
                # the stale result must never execute afterwards.
                continue
            self._results.put({
                "uid": job["uid"],
                "generation": job["generation"],
                "transcript": transcript,
                "error": error,
                "engine": self._engine_label,
                "duration_ms": job["duration_ms"],
                "truncated": job["truncated"],
            })

    def _run_transcribe(self, pcm):
        """Write the utterance to a temporary WAV and transcribe it."""
        wav_path = _write_temp_wav(pcm)
        try:
            return self._transcribe_fn(wav_path)
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass


def _write_temp_wav(pcm):
    """Write 16 kHz mono s16 PCM to a temporary WAV file."""
    handle, path = tempfile.mkstemp(suffix=".wav", prefix="listener_")
    os.close(handle)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    return path


class ResultDrainer:
    """Hands each utterance result to the app thread at most once.

    Lives in st.session_state for the whole Streamlit session, so
    reruns and repeated fragment polling can never execute the same
    action twice. Results from an outdated generation are dropped.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._seen = set()

    def drain(self, worker):
        """Return fresh, current-generation results (once per uid)."""
        if worker is None:
            return []
        results = worker.drain_results()
        fresh = []
        with self._lock:
            current = worker.generation()
            for result in results:
                uid = result.get("uid")
                if not uid or uid in self._seen:
                    continue
                self._seen.add(uid)
                if result.get("generation") != current:
                    continue  # Stale: captured before a reset.
                fresh.append(result)
        return fresh
