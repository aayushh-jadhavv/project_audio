"""Sample-level Ring Buffer module"""

from dataclasses import dataclass
from enum import StrEnum
import logging
from threading import RLock

import numpy as np

class OverflowPolicy(StrEnum):
    """Available policies for overflow buffer"""
    DROP_OLDEST = "drop_oldest"
    RAISE = "raise"

@dataclass(frozen=True)
class BufferStatistics:
    """
    Statistics snapshot of buffer, which is immutable (for debugging mostly).
    """
    samples_written: int
    samples_read: int
    samples_dropped: int
    overflow_count: int
    underflow_count: int

class RingBuffer:
    """
    Store a continuous streams of mono ``float32`` samples.

    Parameters
    ----------
    block_size : int
        Audio callback block size. Does not constrain reads or writes, only used to calculate capacity.
    slots: int
        Number of slots to store in buffer.
    overflow_policy : OverflowPolicy, default=OverflowPolicy.DROP_OLDEST
        Behavior when buffer is overflowed.
    """

    def __init__(self, block_size:int, slots:int, overflow_policy:OverflowPolicy):
        self.capacity = block_size*slots
        self.slots_len = slots
        self.buffer = np.empty(self.capacity, dtype=np.float32)
        self.read_index = 0
        self.write_index = 0
        self.available_samples = 0
        self.overflow_policy = overflow_policy

        self.samples_written = 0
        self.samples_read = 0
        self.samples_dropped = 0
        self.underflow_count = 0
        self.overflow_count = 0

        self._lock = RLock() # reentrant lock, to protect shared state from multiple threads

    def _discard(self, no_of_samples:int) -> None:
        """
        Discard samples from buffer if available. The rlock is held by caller.

        Parameters
        ----------
        no_of_samples : int
        Number of samples to discard.
        """
        self.read_index = (self.read_index + no_of_samples) % self.capacity
        self.available_samples -= no_of_samples
        self.samples_dropped += no_of_samples

    def _copy_from_buffer(self, destination : np.ndarray, no_of_samples : int) -> None:
        # A separate implementation so that we can reuse in both read_into and peek. Hence, this function does not change read_index or available_samples
        """
        Copy samples from the buffer to a destination array, without changing ``available_samples`` and ``read_index``

        Parameters
        ----------
        destination : np.ndarray
            1-D array where samples should be copied in.
        no_of_samples : int
            no of samples to copy.
        """
        remaining = self.capacity - self.read_index
        if no_of_samples > remaining:
            destination[ : remaining] = self.buffer[self.read_index : ]
            destination[remaining : no_of_samples] = self.buffer[ : no_of_samples - remaining]
            return

        destination[ : no_of_samples] = self.buffer[self.read_index : self.read_index + no_of_samples]
        return

    def append(self, samples:np.ndarray) -> None:
        """
        Append samples to the buffer.

        Parameters
        ----------
        samples : np.ndarray
            Array of audio samples.
        """
        samples_f32 = np.asarray(samples, dtype=np.float32)
        if samples_f32.ndim != 1:
            raise ValueError("Samples must be 1D array")
        if samples_f32.size > self.capacity or samples_f32.size <= 0:
            raise BufferError("Invalid sample size")

        no_of_samples = int(samples_f32.size)
        with self._lock:
            overflow_amt = self.available_samples + no_of_samples - self.capacity
            if overflow_amt > 0:
                self.overflow_count += 1

                if self.overflow_policy is OverflowPolicy.RAISE:
                    raise BufferError("Ring buffer overflowed")

                logging.debug("Ring buffer overflowed. Dropping oldest samples.")
                self._discard(overflow_amt)

            remaining = self.capacity - self.write_index
            if remaining >= no_of_samples:
                self.buffer[self.write_index : self.write_index + no_of_samples] = samples_f32
            else:
                self.buffer[self.write_index:] = samples_f32[:remaining]
                self.buffer[: no_of_samples - remaining] = samples_f32[remaining:]

            self.write_index = (self.write_index + no_of_samples) % self.capacity
            self.available_samples += no_of_samples
            self.samples_written += no_of_samples

    def read_into(self, destination:np.ndarray) -> int:
        """
        Copy directly into a pre_allocated ``destination`` array.
        Parameters
        -----------
        destination : np.ndarray
            Pre-allocated 1-D array into where samples should be read.

        Returns
        -------
        int
            Number of samples which are copied.
        """
        output = np.asarray(destination)
        if output.ndim != 1:
            raise ValueError("Destination array must be one-dimensional")
        if output.dtype != np.float32:
            raise ValueError("Destination array must have dtype as float32")

        no_of_samples = int(output.size)
        if no_of_samples == 0:
            return 0

        with self._lock:
            if no_of_samples > self.available_samples:
                self.underflow_count += 1
                logging.debug("Buffer Underflow")
                raise BufferError("Too many samples requested")

            self._copy_from_buffer(destination, no_of_samples)
            self.read_index = (self.read_index + no_of_samples) % self.capacity
            self.available_samples -= no_of_samples
            self.samples_read += no_of_samples
            return no_of_samples

    def read(self, no_of_samples : int) -> np.ndarray:
        """
        Read and remove samples from the buffer.

        Parameters
        ----------
        no_of_samples : int
            How many samples to read and remove.

        Returns
        -------
        np.ndarray
            A new ``float32`` array. New so that this array does not alias the internal buffer storage.
        """
        if not isinstance(no_of_samples, int):
            raise TypeError("no_of_samples should be an int")
        if no_of_samples < 0:
            raise ValueError("no_of_samples should not be negative.")
        if no_of_samples == 0:
            return np.empty(0, dtype=np.float32)

        result = np.empty(no_of_samples, dtype=np.float32)
        self.read_into(result)
        return result

    def peek(self, no_of_samples : int) -> np.ndarray:
        """
        Return samples from the buffer without modifying read_index.

        Returns
        -------
        no_of_samples: int
            How many samples to read from the buffer.

        Returns
        -------
        np.ndarray
            A new ``float32`` array. New so that this array does not alias the internal buffer storage.
        """
        if not isinstance(no_of_samples, int):
            raise TypeError("no_of_samples should be an int")
        if no_of_samples < 0:
            raise ValueError("no_of_samples should not be negative.")
        if no_of_samples == 0:
            return np.empty(0, dtype=np.float32)

        with self._lock:
            if no_of_samples > self.available_samples:
                self.underflow_count += 1
                logging.debug("Buffer Underflow")
                raise BufferError("Too many samples requested")

            result = np.empty(no_of_samples, dtype=np.float32)
            self._copy_from_buffer(result, no_of_samples)
            return result

    def available_to_read(self) -> int:
        """
        Return the number of samples currently stored in the buffer.

        Returns
        -------
        int
            Number of samples available for reading.
        """
        with self._lock: # why is there lock?
            return self.available_samples

    def available_to_write(self) -> int:
        """
        Return the number of samples that can be stored.

        Returns
        -------
        int
            Number of available space to read samples.
        """
        with self._lock: # why is there lock?
            return self.capacity - self.available_samples

    def clear(self) -> None:
        """
        Remove all samples in the buffer but, preserve lifetime statistics.
        """
        with self._lock:
            self.read_index = 0
            self.write_index = 0
            self.available_samples = 0
        logging.debug("Buffer cleared")

    @property
    def statistics(self) -> BufferStatistics:
        """
        Return an immutable snapshot of the buffer's statistics.

        Returns
        -------
        BufferStatistics
            Immutable snapshot of the buffer's statistics.
        """
        # Lock ensures all five fields come from the same moment
        with self._lock:
            return BufferStatistics(
                overflow_count=self.overflow_count,
                underflow_count=self.underflow_count,
                samples_read=self.samples_read,
                samples_dropped=self.samples_dropped,
                samples_written=self.samples_written
            )

    def reset_statistics(self) -> None:
        """
        Reset all buffer statistics.
        """
        with self._lock:
            self.overflow_count = 0
            self.underflow_count = 0
            self.samples_read = 0
            self.samples_dropped = 0
            self.samples_written = 0

    def is_empty(self) -> bool:
        """
        Check if the buffer is empty.

        Returns
        -------
        bool
         ``True`` if the buffer is empty.
        """
        return self.available_samples == 0

    def is_full(self) -> bool:
        """
        Check if the buffer is full.

        Returns
        -------
        bool
         ``True`` if the buffer is full.
        """
        return self.available_samples == self.capacity