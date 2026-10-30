"""Low-latency speaker playback backed by a sample-level ring buffer."""

from collections.abc import Callable
import logging
from typing import Any

import numpy as np
import sounddevice as sd

from src.audio.ring_buffer import RingBuffer, OverflowPolicy

class AudioOutput:
    """
    Feed processed mono samples to a SoundDevice output callback
    """
    def __init__(self,
                 sr : float,
                 channels : int = 1,
                 blocksize : int = 1024,
                 device : int | str | None = None,
                 buffer_slots : int = 16,
                 stream_factory : Callable[..., Any] | None = None):
        if sr <= 0:
            raise ValueError("Sample Rate (sr) must be greater than 0")
        if channels <= 0:
            raise ValueError("Channels must be greater than 0")
        if blocksize <= 0:
            raise ValueError("Block Size must be greater than 0")

        self.sr = float(sr)
        self.channels = int(channels)
        self.blocksize = int(blocksize)
        self.device = device
        self.buffer = RingBuffer(self.blocksize, buffer_slots, OverflowPolicy.DROP_OLDEST)

        self.callback_count = 0
        self.status_count = 0
        self.underrun_count = 0
        self._stream_factory = stream_factory or sd.OutputStream
        self.stream: Any | None = None

    def _create_stream(self) -> Any:
        """
        Create the configured output stream.
        """
        return self._stream_factory(
            samplerate=self.sr,
            channels=self.channels,
            blocksize=self.blocksize,
            dtype="float32",
            device=self.device,
            callback=self.callback
        )

    def start(self) -> None:
        """
        Create and start the output stream.
        """
        if self.stream in None:
            self.stream = self._create_stream()
        if not bool(getattr(self.stream, 'active', False)):
            self.stream.start()

    def pause(self) -> None:
        """
        Pause playback without disposing the stream.
        """
        if self.stream is not None and bool(getattr(self.stream, 'active', False)):
            self.stream.stop()

    def stop(self) -> None:
        """
        Stop and close the output stream.
        """
        if self.stream is None:
            return
        if bool(getattr(self.stream, 'active', False)):
            self.stream.stop()
        self.stream.close()
        self.stream = None

    def write(self, samples : np.ndarray) -> None:
        """
        Queue processed mono samples for playback.
        """
        self.buffer.append(np.asarray(samples, dtype=np.float32))

    def callback(self,
                 outdata : np.ndarray,
                 frames : int,
                 time_info : Any,
                 status : Any) -> None:
        """
        Fill an output callback block, padding missing samples with silence.

        Parameters
        ----------
        outdata : np.ndarray
            Output buffer which should be filled with the audio data.
        frames : int
            Number of audio frames the callback needs to process.
        time_info : Any
            An object providing timing timestamps
        status : Any
            A flag object that is ``False`` or ``None`` if everything is normal, but acts truthy if a warning or error flag occurs.
        """
        del time_info
        if status:
            self.status_count += 1
            logging.warning("AudioOutput status: %s", status)

        outdata.fill(0.0)
        available = min(frames, self.buffer.available_to_read())
        if available:
            samples = self.buffer.read(available)
            if outdata.ndim == 1:
                outdata[:available] = samples
            else:
                outdata[:available, :] = samples[:, np.newaxis]
        if available < frames:
            self.underrun_count += 1

        self.callback_count += 1

    def __enter__(self) -> "AudioOutput":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()