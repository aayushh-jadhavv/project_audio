"""Reusable real-valued FFT processor"""
from dataclasses import dataclass

import numpy as np
from scipy import fft

@dataclass(frozen=True)
class FFTResult:
    """Frequency-domain representation of one real-valued frame."""

    frequencies: np.ndarray
    complex_spectrum: np.ndarray
    magnitudes: np.ndarray
    magnitudes_db: np.ndarray
    phase: np.ndarray

class FFTProcessor:
    """
    Compute a normalized one-sided FFT for fixed-size frames.

    Parameters
    ----------
    sample_rate : float
        Sample rate of the audio.
    frame_size : int
        Size of a frame to process.
    floor_db : float, default=-160.0
        Lowest magnitude of audio to consider.
    """
    def __init__(
            self,
            sample_rate:float,
            frame_size : int,
            floor_db : float = -160.0
    ):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be greater than 0")
        if frame_size <= 0:
            raise ValueError("frame_size must be greater than 0")
        if floor_db >= 0:
            raise ValueError("floor_db must be less than 0")

        self.sr = float(sample_rate)
        self.frame_size = int(frame_size)
        self.floor_db = float(floor_db)
        self._freq_bins = fft.rfftfreq(self.frame_size, d=1.0/self.sr)
        self._freq_bins.flags.writeable = False

    @property
    def frequencies(self) -> np.ndarray:
        """
        Return the read-only frequency-bin array.

        Returns
        -------
        np.ndarray
            Frequency bin array (in Hz).
        """
        return self._freq_bins

    def process(self, frame : np.ndarray) -> FFTResult:
        """
        Compute frequency bins, magnitude, dB magnitude and phase.

        Parameters
        ----------
        frame : np.ndarray
            Audio frame to compute.

        Returns
        -------
        FFTResult
            Result of FFT
        """
        frame_f32 = np.asarray(frame, dtype=np.float32)
        if frame_f32.ndim != 1:
            raise ValueError("frame must be one-dimensional")
        if frame_f32.shape[0] != self.frame_size:
            raise ValueError(f"frame length mismatch: expected {self.frame_size}, got {frame_f32.shape[0]}")

        complex_spectrum = fft.rfft(frame_f32)
        magnitudes = np.abs(complex_spectrum) / self.frame_size
        if self.frame_size > 1:
            if self.frame_size % 2 == 0:
                magnitudes[1:-1] *= 2.0
            else:
                magnitudes[1:] *= 2.0

        floor_linear = 10.0 ** (self.floor_db / 20.0)
        magnitudes_db = 20.0 * np.log10(np.maximum(magnitudes, floor_linear))
        phases = np.angle(complex_spectrum).astype(np.float32)

        return FFTResult(
            frequencies=self._freq_bins,
            complex_spectrum=complex_spectrum,
            magnitudes=magnitudes,
            magnitudes_db=magnitudes_db,
            phase=phases
        )