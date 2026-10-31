"""Worker-thread runtime connecting audio callbacks to the DSP processor"""

from dataclasses import dataclass
import logging
from threading import RLock, Event, Thread, current_thread
from time import perf_counter

from src.audio.capture import AudioInput
from src.audio.playback import AudioOutput
from src.dsp.engine import DSPProcessor, DSPSettings, ProcessingResult

@dataclass(frozen=True)
class EngineStatus:
    """Immutable runtime diagnostics displayed by the GUI"""
    running: bool
    paused: bool
    blocks_processed: int
    processing_overruns: int
    last_processing_seconds: float

class AudioEngine:
    """
    Coordinate capture, DSP and playback without blocking an audio callback.

    The SoundDevice callbacks only copy data between driver buffers and ring buffers. A dedicted worker thread performs all filtering and analysis.

    Parameters
    ----------
    dsp_settings : DSPSettings
        Settings for DSP processing
    block_size : int, default=1024
        Size of block for audio callback and DSP processing
    input_device : int | str | None, default=None
        Device for audio input
    output_device : int | str | None, default=None
        Device for audio output
    processor : DSPProcessor | None, default=None
        DSP processor
    audio_input : AudioInput | None, default=None
        Audio capture stream
    audio_output : AudioOutput | None, default=None
        Audio playback stream
    """

    def __init__(
        self,
        dsp_settings: int = 1024,
        block_size: int | str | None = None,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
        processor: DSPProcessor | None = None,
        audio_input: AudioInput | None = None,
        audio_output: AudioOutput | None = None
    ):
        if block_size <= 0:
            raise ValueError("block_size must be greater than 0")

        self.dsp_settings = dsp_settings
        self.block_size = int(block_size)
        self.processor = processor or DSPProcessor(dsp_settings)
        self._wake_event = Event()
        self.input = audio_input or AudioInput(
            sr=dsp_settings.sample_rate,
            blocksize=self.block_size,
            device=input_device,
            on_samples_available=self._wake_event.set
        )
        self.output = audio_output or AudioOutput(
            sr=dsp_settings.sample_rate,
            blocksize=self.block_size,
            device=output_device
        )

        self._running = False
        self._paused = False
        self._worker: Thread | None = None
        self._latest_result: ProcessingResult | None = None
        self._blocks_processed = 0
        self._processing_overruns = 0
        self._last_processing_seconds = 0.0
        self._lock = RLock()

    def _processing_loop(self) -> None:
        """
        Consume captured blocks until :meth:`stop` is called.
        """
        block_duration = self.block_size / self.dsp_settings.sample_rate
        while True:
            with self._lock:
                running = self._running
                paused = self._paused
            if not running:
                return
            if paused:
                self._wake_event.wait(timeout=0.05)
                self._wake_event.clear()
                continue

            self._wake_event.wait(timeout=0.02)
            self._wake_event.clear()
            while self.input.buffer.available_to_read() >= self.block_size:
                with self._lock:
                    if not self._running or self._paused:
                        break
                started = perf_counter()
                raw = self.input.buffer.read(self.block_size)
                result = self.processor.process_block(raw)
                self.output.write(result.processed_frame)
                elapsed = perf_counter() - started

                with self._lock:
                    self._latest_result =result
                    self._blocks_processed += 1
                    self._last_processing_seconds = elapsed
                    if elapsed > block_duration:
                        self._processing_overruns += 1

    def start(self) -> None:
        """
        Start or resume the complete real-time pipeline.
        """
        with self._lock:
            if self._running:
                if self._paused:
                    self.output.start()
                    self.input.start()
                    self._paused = False
                    self._wake_event.set()
                return

            self.input.buffer.clear()
            self.output.buffer.clear()
            self.processor.reset()
            self._running = True
            self._paused = False
            self._worker = Thread(
                target=self._processing_loop,
                name="audio-dsp-worker",
                daemon=True
            )
            self._worker.start()
        try:
            self.output.start()
            self.input.start()
        except Exception:
            logging.exception("Unable to start the audio streams")
            self.stop()
            raise

    def pause(self) -> None:
        """
        Pause both streams while keeping worker and devices reusable.
        """
        with self._lock:
            if not self._running or self._paused:
                return
            self._paused = True
        self.input.pause()
        self.output.pause()

    def stop(self) -> None:
        """
        Stop streams and terminate the DSP worker.
        """
        with self._lock:
            was_running = self._running
            self._running = False
            self._paused = False
            worker = self._worker
            self._worker = None
            self._wake_event.set()

        self.input.stop()
        self.output.stop()
        if was_running and worker is not None and worker is not current_thread():
            worker.join(timeout=2.0)

    @property
    def latest_result(self) -> ProcessingResult | None:
        """
        Return the latest immutable processing-result reference.
        """
        with self._lock:
            return self._latest_result

    @property
    def status(self) -> EngineStatus:
        """
        Return current runtime diagnostics.
        """
        with self._lock:
            return EngineStatus(
                running=self._running,
                paused=self._paused,
                blocks_processed=self._blocks_processed,
                processing_overruns=self._processing_overruns,
                last_processing_seconds=self._last_processing_seconds
            )

    def __enter__(self) -> "AudioEngine":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()