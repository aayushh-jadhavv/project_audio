"""Build fixed-size overlapping analysis frames from a sample ring buffer."""
from typing import Optional

import numpy as np

from src.audio.ring_buffer import RingBuffer

class FrameBuilder:
    """
    Extract overlapping frames without reallocating the worker frame.

    Parameters
    ----------
    rb : RingBuffer
        Sample RingBuffer with audio samples.
    frame_size : int
        Size of the frame to build.
    hop_size : int
        At what interval should the frames be build. This is for how much overlap should exist.
    """
    def __init__(
            self,
            rb : RingBuffer,
            frame_size : int,
            hop_size : int
    ):
        if frame_size <= 0:
            raise ValueError("frame_size should be greater than 0")
        if frame_size > rb.capacity:
            raise ValueError("frame_size should be within the Ring Buffer's capacity")
        if hop_size <= 0:
            raise ValueError("hop_size should be greater than 0")
        if hop_size > frame_size:
            raise ValueError("hop_size should be less than frame_size")
        if frame_size & (frame_size - 1):
            raise ValueError("frame_size should be in powers of 2")

        self.buffer = rb
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.frame = np.empty(self.frame_size, dtype=np.float32)
        self._bootstrapped = False

    def are_frames_available(self) -> bool:
        """
        Check whether there are enough samples to build a frame

        Returns
        -------
        bool
            ``True`` if enough samples are available, ``False`` otherwise.
        """
        required_samples = self.frame_size if self._bootstrapped else self.hop_size
        return self.buffer.available_to_read() >= required_samples

    def next_frame(self) -> Optional[np.ndarray]:
        """
        Return the next analysis frame when enough samples are available.
        
        Returns
        -------
        np.ndarray | None
            Internal reusable frame, or ``None`` while more samples are needed.
            The caller must treat the returned array as read-only and copy it before retaining it.
        """
        if not self.are_frames_available():
            return None

        if not self._bootstrapped:
            self.frame[ : ] = self.buffer.read(self.frame_size)
            self._bootstrapped = True
            return self.frame

        overlap = self.frame_size - self.hop_size
        if overlap:
            self.frame[ : overlap] = self.frame[self.hop_size : ]
        self.frame[overlap : ] = self.buffer.read(self.frame_size)
        return self.frame

    def reset(self) -> None:
        """
        Forget the overlap carried from the previous frame.
        """
        self._bootstrapped = False