"""Callback-based microphone capture module."""
# Callable is an object that can be called, like a function. Callable[[int], str] means a function which accepts an int argument and returns a str.
from collections.abc import Callable
from typing import Any
import logging

import numpy as np
import sounddevice as sd

from src.audio.ring_buffer import RingBuffer, OverflowPolicy

class AudioInput:
    """
    Capture microphone samples into a sample-level :class:`RingBuffer`.

    Stream creation is delayed until :meth:`start`, which keeps construction and unit testing separate from the presence of audio hardware.
    """
    def __init__(self,
                 sr : float,
                 channels : int = 1,
                 blocksize : int = 1024,
                 device : int | str | None = None,
                 buffer_slots : int = 16,
                 overflow_policy : OverflowPolicy = OverflowPolicy.DROP_OLDEST,
                 # This will be a function that should be invoked whenever new data may be available.
                 on_samples_available : Callable[[], None] | None = None,
                 # This will be used in testing (take a lock here again while testing)
                 stream_factory : Callable[..., Any] | None = None
                 ):
        if sr <= 0:
            raise ValueError("Sample Rate (sr) must be greater than 0")
        if channels <= 0:
            raise ValueError("Channels must be greater than 0")
        if blocksize <= 0:
            raise ValueError("Block size must be greater than 0")

        self.sr = float(sr)
        self.channels = int(channels)
        self.blocksize = int(blocksize)
        self.device = device
        self.buffer = RingBuffer(self.blocksize, buffer_slots, overflow_policy)

        self.callback_count = 0
        self.status_count = 0
        self._on_samples_available = on_samples_available
        self._stream_factory = stream_factory or sd.InputStream
        self.stream: Any | None = None

    def _create_stream(self) -> Any:
        """
        Create the configured input stream
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
        Create and start the input stream.
        """
        if self.stream is None:
            self.stream = self._create_stream()
        # if the stream isn't active, active it.
        # getattr(x, 'y', ...) is basically getting the value of x.y. The third argument is default value to consider, if the attribute isn't found.
        if not bool(getattr(self.stream, 'active', False)):
            self.stream.start()

    def pause(self) -> None:
        """
        Pause capture without disposing the stream.
        """
        if self.stream is not None and bool(getattr(self.stream, 'active', False)):
            self.stream.stop()

    def stop(self) -> None:
        """
        Stop and close the input stream
        """
        if self.stream is None:
            return
        if bool(getattr(self.stream, 'active', False)):
            self.stream.stop()
        self.stream.close()
        self.stream = None

    def callback(self,
                 indata : np.ndarray,
                 frames : int,
                 time_info : Any,
                 status : Any) -> None:
        """
        Copy callback samples into the ring buffer.

        Parameters
        ----------
        indata : np.ndarray
            An 1-D array containing recorded audio samples. Shape of this array is ``(frames, channels)``.
        frames : int
            Number of audio frames in the current block.
        time_info : Any
            An object providing timing timestamps.
        status : Any
            A flag object that is ``False`` or ``None`` if everything is normal, but acts truthy if a warning or error flag occurs.
        """
        # deleting these local variables, since we don't need them.
        del frames, time_info

        if status:
            self.status_count += 1
            logging.warning("AudioInput status: %s", status)

        data = np.asarray(indata, dtype=np.float32)
        if data.ndim == 1:
            samples = data
        elif data.ndim == 2 and data.shape[1] == 1:
            samples = data[:, 0] # data was 2-dimensional (like [[sample0], [sample1], [sample2], ...]), samples is [sample0, sample1, sample2, ...]
        elif data.ndim == 2:
            samples = np.mean(data, axis=1, dtype=np.float32)
        else:
            logging.error("Ignoring callback data with shape %s", data.shape)
            return

        try:
            self.buffer.append(samples)
        except BufferError:
            logging.exception("Audio input samples could not be buffered.")
            return

        self.callback_count += 1
        if self._on_samples_available is not None:
            self._on_samples_available()

    def read_frame(self) -> np.ndarray:
        """
        Read one nominal block from the buffer.

        Returns
        -------
        np.ndarray
            Array containing samples from the buffer.
        """
        return self.buffer.read(self.blocksize)

    @staticmethod
    def query_devices() -> Any:
        """
        Return devices reported by SoundDevice
        """
        return sd.query_devices()

    def __enter__(self) -> "AudioInput": # "AudioInput" is forward reference to the class currently being defined.
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()