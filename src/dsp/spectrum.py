"""Spectrum features derived from an :class:`FFTResult`"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from matplotlib.bezier import find_bezier_t_intersecting_with_closedpath
from pygments.styles import manni
from scipy import signal

from src.dsp.fft import FFTResult

@dataclass(frozen=True)
class SpectrumResult:
    """Magnitude spectrum and its most useful descriptive features"""
    frequencies: np.ndarray
    magnitudes: np.ndarray
    magnitudes_db: np.ndarray
    peak_frequencies: np.ndarray
    peak_magnitudes: np.ndarray
    peak_magnitudes_db: np.ndarray
    centroid: float
    noise_floor: float
    dynamic_range: float

@dataclass(frozen=True)
class SpectrumSettings:
    """
    Configuration settings for deriving meaningful spectral peaks

    Note
    ----
    if `max_peaks` is None, all the peaks will be returned.
    """
    max_peaks: int | None = 5
    peak_prominence_db: float | None = 6.0

class SpectrumAnalyzer:
    """
    Calculate peaks, centroid, noise floor and dynamic range

    Parameters
    ----------
    settings : SpectrumSettings
        Settings for Spectrum Analysis (peak detection)
    """
    def __init__(self, settings: SpectrumSettings | None = None):
        self.settings = settings or SpectrumSettings()
        if self.settings is not None and self.settings.max_peaks <= 0:
            raise ValueError("max_peaks must be greater than 0")

    def _find_peaks(
            self,
            n: int | None,
            freq_bins: np.ndarray,
            magnitudes: np.ndarray,
            magnitudes_db:np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return spectral peaks, ordered from strongest to weakest

        Parameters
        ----------
        n : int | None
            Number of spectral peaks to return. ``None`` means all detected peaks should be returned.
        freq_bins : np.ndarray
            Array of Frequency Bins
        magnitudes : np.ndarray
            Array of magnitudes (in linear form)
        magnitudes_db : np.ndarray
            Array of magnitudes (in dB)

        (see about returns)
        """
        prominence = self.settings.peak_prominence_db
        kwargs = {} if prominence is None else {"prominence": prominence}
        peak_indices, _ = signal.find_peaks(magnitudes_db, **kwargs)
        if peak_indices.size == 0:
            empty = np.empty(0, dtype=float)
            return empty, empty.copy(), empty.copy()

        order = np.argsort(magnitudes_db[peak_indices])[::-1] # [::-1] reverses the argsort order
        if n is not None:
            order = order[:n]
        selected = peak_indices[order]
        return(
            freq_bins[selected],
            magnitudes[selected],
            magnitudes_db[selected]
        )

    def _compute_centroid(self, freq_bins: np.ndarray, magnitudes: np.ndarray) -> float:
        """
        Compute the spectral centroid in Hz.

        Parameters
        ----------
        freq_bins : np.ndarray
            Array of frequency bins.
        magnitudes : np.ndarray
            Array of magnitudes (in linear form)

        Returns
        -------
        float
            The Spectral centroid (in Hz).
        """
        denominator = float(np.sum(magnitudes))
        if denominator <= 0.0:
            return float('nan')
        return float(np.sum(freq_bins * magnitudes) / denominator)

    def _estimate_noise_floor(self, magnitudes_db:np.ndarray) -> float:
        """
        Estimate noise floor of magnitudes, after excluding .

        Parameters
        ----------
        magnitudes_db : np.ndarray
            Array of magnitudes (in dB).

        Returns
        -------
        float
            Estimated noise floor.
        """
        finite = magnitudes_db[np.isfinite(magnitudes_db)]
        if finite.size == 0:
            return float('nan')
        if finite.size <= 5:
            return float(np.percentile(finite, 10))

        keep_count = finite.size - min(5, finite.size - 1)
        background = np.partition(finite, keep_count-1)[:keep_count] # see this again
        return float(np.percentile(background, 10))

    def _compute_dynamic_range(self, magnitudes_db: np.ndarray, noise_floor: float) -> float:
        """
        Compute the distance between the strongest bin and noise floor.

        Parameters
        ----------
        magnitudes_db : np.ndarray
            Array of magnitudes (in dB).
        noise_floor : float
            The noise floor of the spectrum.

        Returns
        -------
        float :
            Distance between the strongest bin and noise floor (dynamic range).
        """
        finite = magnitudes_db[np.isfinite(magnitudes_db)]
        if finite.size == 0 or not np.isfinite(noise_floor):
            return float('nan')
        return float(np.max(finite) - noise_floor)

    def analyze(self, fftResult: FFTResult) -> SpectrumResult:
        """
        Analyze one FFT Result at a time.

        Parameters
        ----------
        fftResult : FFTResult
            Result of Fast Fourier Transform

        Returns
        -------
        SpectrumResult
            Spectral analysis result
        """
        freqs = fftResult.frequencies
        magnitudes = fftResult.magnitudes
        magnitudes_db = fftResult.magnitudes_db

        peak_freqs, peaks, peaks_db = self._find_peaks(self.settings.max_peaks, freqs, magnitudes, magnitudes_db)
        centroid = self._compute_centroid(freqs, magnitudes)
        noise_floor = self._estimate_noise_floor(magnitudes_db)
        dynamic_range = self._compute_dynamic_range(magnitudes_db, noise_floor)

        return SpectrumResult(
            frequencies=freqs,
            magnitudes=magnitudes,
            magnitudes_db=magnitudes_db,
            peak_frequencies=peak_freqs,
            peak_magnitudes=peaks,
            peak_magnitudes_db=peaks_db,
            centroid=centroid,
            noise_floor=noise_floor,
            dynamic_range=dynamic_range
        )
