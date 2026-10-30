"""Cascaded parametric equalizer built from peaking biquad filters."""
from dataclasses import dataclass
from threading import RLock

import numpy as np

from src.dsp.filters import BiquadFilter, design_peaking_eq

@dataclass(frozen=True)
class EqualizerBand:
    """Configuration for one parametric peaking band."""

    center_frequency: float
    gain_db: float = 0.0
    q: float= 1.0
    enabled: bool = True

DEFAULT_BAND_FREQUENCIES = (60.0, 250.0, 1000.0, 4000.0, 12000.0)

class ParametricEqualizer:
    """
    Process audio through a cascade of independently adjustable bands.

    Parameters
    ----------
    sample_rate : float
        Sample rate of the audio
    bands : list[EqualizerBand] | tuple[EqualizerBand, ...] | None, default=None
        Frequencies for peaking equalizer, default is just None, when the default frequencies are considered - 60Hz, 250Hz, 1000Hz, 4000Hz & 12000Hz
    """
    def __init__(
            self,
            sample_rate: float,
            bands: list[EqualizerBand] | tuple[EqualizerBand, ...] | None = None
    ):
        if sample_rate <= 0:
            raise ValueError("sample_rate should be greater than 0")

        self.sr = float(sample_rate)
        configured = (
            [
                EqualizerBand(freq)
                for freq in DEFAULT_BAND_FREQUENCIES
                if freq < sample_rate / 2
            ]
            if bands is None
            else list(bands)
        )
        self.bypassed = False
        self._bands = [self._validate_band(band) for band in configured]
        self._filters = [self._create_filter(band) for band in self._bands]
        self._lock = RLock

    @property
    def bands(self) -> tuple[EqualizerBand, ...]:
        """Return an immutable snapshot of all band settings."""
        with self._lock:
            return tuple(self._bands)

    def _validate_band(self, band: EqualizerBand) -> EqualizerBand:
        """
        Check if a passed band is appropriate.

        Parameters
        ----------
        band : EqualizerBand
            Equalizer band to be validated

        Returns
        -------
        EqualizerBand
            Validated equalizer band
        """
        if not 0 < band.center_frequency < self.sr / 2:
            raise ValueError("band center_frequency must be between 0 and Nyquist")
        if band.q <= 0:
            raise ValueError("band q must be greater than 0")
        if not -36.0 <= band.gain_db <= 36.0:
            raise ValueError("band gain_db must be between -36 and 36")
        return band

    def _create_filter(self, band: EqualizerBand) -> BiquadFilter:
        """
        Create filter for an equalizer band.

        Parameters
        ----------
        band : EqualizerBand
            
        """

    def update_band(self):
        pass

    def process(self):
        pass

    def frequency_response(self):
        pass

    def reset(self):
        pass