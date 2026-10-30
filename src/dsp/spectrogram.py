"""Rolling STFT history used by GUI plot."""
from dataclasses import dataclass

import numpy as np

from src.dsp.spectrum import SpectrumResult

@dataclass(frozen=True)
class SpectrogramResult:
    """
    Immutable snapshot of rolling spectral history.
    """
    frequencies: np.ndarray
    times: np.ndarray
    magnitudes_db: np.ndarray

class SpectrogramHistory:
    """
    Maintain a fixed number of consecutive spectrum columns

    Parameters
    ----------
    sample_rate : float
        Sample rate of Audio Input
    frame_size : int
        Size of audio frames
    hop_size : int
        Intervals over which audio should be read
    history_size : int, default=200
        How much of history should be shown at a time
    floor_db : float, default=-120.0
        Noise floor from spectrum analysis
    """
    def __init__(
        self,
        sample_rate: float,
        frame_size: int,
        hop_size: int,
        history_size: int = 200,
        floor_db: float = -120.0
    ):
        if sample_rate <= 0:
            raise ValueError("sample_rate should be greater than 0")
        if frame_size <= 0:
            raise ValueError("frame_size should be greater than 0")
        if hop_size <= 0 or hop_size > frame_size:
            raise ValueError("hop_size should be within limits (0 < hop_size <= frame_size)")
        if history_size <= 0:
            raise ValueError("history_size should be greater than 0")

        self.sr = float(sample_rate)
        self.frame_size = int(frame_size)
        self.hop_size = int(hop_size)
        self.history_size = int(history_size)
        self.floor_db = float(floor_db)

        self._frequencies = np.fft.rfftfreq(self.frame_size, d=1.0 / self.sr)
        self._history = np.full((self._frequencies.size, self.frame_size), fill_value=self.floor_db, dtype=np.float32)
        self._columns_written = 0
        
    def update(self, spectrum: SpectrumResult) -> SpectrogramResult:
        """
        Append one spectrum and return the updated history snapshot.

        Parameters
        ----------
        spectrum : SpectrumResult
            Result of spectrum analysis

        Returns
        -------
        SpectrogramResult
            Result of Spectrogram Analysis
        """
        if spectrum.magnitudes_db.shape != self._frequencies.shape:
            raise ValueError("magnitude bin should have same shape as frequency bin")

        self._history[:, :-1] = self._history[:, 1:]
        self._history[:, -1] = np.maximum(spectrum.magnitudes_db, self.floor_db)
        self._columns_written += 1
        return self.snapshot()

    def snapshot(self) -> SpectrogramResult:
        """
        Return a defensive copy of the full rolling history
        """
        column_seconds = self.hop_size / self.sr
        times = (
            np.arange(self.history_size, dtype=float) - self.history_size + 1
        ) * column_seconds
        return SpectrogramResult(
            frequencies=self._frequencies.copy(),
            times=times,
            magnitudes_db=self._history.copy()
        )

    def clear(self) -> None:
        """
        Reset every history column to the configured noise floor
        """
        self._history.fill(self.floor_db)
        self._columns_written = 0

    @property
    def columns_written(self) -> int:
        """Return the number of spectra received since the last clear."""
        # This is used instead of making _columns_written public, since this is a read-only interface.
        return self._columns_written