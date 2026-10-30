"""Biquad filters based on the RBJ Audio EQ Cookbook"""
from dataclasses import dataclass

import numpy as np
from scipy import signal

@dataclass(frozen=True)
class BiquadCoefficients:
    """Normalized second-order IIR coefficients where ``a0`` is one."""

    b0: float
    b1: float
    b2: float
    a1: float
    a2: float

    @property
    def numerator(self) -> np.ndarray:
        """Return numerator coefficients in SciPy order."""
        return np.array([self.b0, self.b1, self.b2], dtype=float)

    @property
    def denominator(self) -> np.ndarray:
        """Return denominator coefficients in SciPy order."""
        return np.array([1.0, self.a1, self.a2], dtype=float)

def design_peaking_eq(
        sample_rate:float,
        center_freq:float,
        gain_db:float,
        q:float
) -> BiquadCoefficients:
    """
    Design a constant-Q peaking equalizer using RBJ equations.

    Parameters
    ----------
    sample_rate : float
        Sample rate of the audio.
    center_freq : float
        Frequency at which boost/cut should happen (f0).
    gain_db : float
        How much boost/cut should happen at the center frequency.
    q : float
        Quality Factor (q = f0/B, where f0 is center frequency and B is frequency bandwidth) denotes sharpness of the boost/cut.

    Return
    -------
    BiquadCoefficients
        Normalized second-order coefficients for the Peaking Equalizer
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate should be greater than 0")
    if not 0 < center_freq < sample_rate/2:
        raise ValueError("center_freq should be within the limits of (0, sample_rate/2)")
    if q <= 0:
        raise ValueError("q should be greater than 0")

    amplitude = 10.0 ** (gain_db / 40.0)
    omega = 2.0 * np.pi * (center_freq / sample_rate)
    alpha = np.sin(omega) / (2.0 * q)
    cosine = np.cos(omega)

    b0 = 1.0 + alpha * amplitude
    b1 = -2.0 * cosine
    b2 = 1.0 - alpha * amplitude
    a0 = 1.0 + alpha / amplitude
    a1 = -2.0 * cosine
    a2 = 1.0 - alpha / amplitude

    return BiquadCoefficients(
        b0 = float(b0 / a0),
        b1 = float(b1 / a0),
        b2 = float(b2 / a0),
        a1 = float(a1 / a0),
        a2 = float(a2 / a0)
    )

class BiquadFilter:
    """Stateful mono biquad suitable for processing consecutive blocks."""

    def __init__(self, coefficients: BiquadCoefficients):
        self.coefficients = coefficients
        self._state = np.zeros(2, dtype=float)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Filter a one-dimensional frame while preserving IIR state.

        Parameters
        ----------
        frame : np.ndarray
            Audio frame to filter.

        Returns
        -------
        np.ndarray
            Filtered audio frame.
        """
        frame_f32 = np.asarray(frame, dtype=np.float32)
        if frame_f32.ndim != 1:
            raise ValueError("frame must be one-dimensional")
        if frame_f32.size == 0:
            return frame_f32.copy()

        output, self._state = signal.lfilter(
            b=self.coefficients.numerator,
            a=self.coefficients.denominator,
            x=frame_f32,
            zi=self._state,
        )
        return output.astype(np.float32)

    def update_coefficients(self, coefficients: BiquadCoefficients, reset_state: bool = False) -> None:
        """
        Update the Filter Coefficients, optionally clearing delay state.

        Parameters
        ----------
        coefficients : BiquadCoefficients
            Filter coefficients to update.
        reset_state : bool, default=False
            Whether to clear delay state, or not.
        """
        self.coefficients = coefficients
        if reset_state:
            self.reset()

    def reset(self) -> None:
        """
        Clear the two filter-delay values.
        """
        self._state.fill(0.0)

    def frequency_response(self, frequencies: np.ndarray, sample_rate: float) -> np.ndarray:
        """
        Return the complex response at selected frequencies.

        Parameters
        ----------
        frequencies : np.ndarray
            Selected frequencies, whose complex response is required.
        sample_rate : float
            Sample rate of the audio input

        Returns
        -------
        np.ndarray
            Array of complex response at the selected frequencies
        """
        _, response = signal.freqz(
            self.coefficients.numerator,
            self.coefficients.denominator,
            worN=np.asarray(frequencies, dtype=float),
            fs=sample_rate)
        return response