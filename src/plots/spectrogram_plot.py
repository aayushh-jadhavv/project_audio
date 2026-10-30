"""High-Performance PyQtGraph renderer for rolling spectrogram results"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg
from mpl_toolkits.axisartist import floating_axes

from src.dsp.spectrogram import SpectrogramResult

@dataclass(frozen=True)
class SpectrogramSettings:
    """
    Apperance settings for a spectrogram image
    """
    color_map: str = "magma"
    floor_db: float = -120
    ceiling_db: float = 0.0

class SpectrogramPlot(pg.PlotWidget):
    """
    Display and incrementally update a scrolling time-frequency image
    """
    def __init__(self,
                 settings: SpectrogramSettings | None = None,
                 parent: QtWidgets.QWidget | None = None):
        self.settings = settings or SpectrogramSettings()
        if self.settings.floor_db >= self.settings.ceiling_db:
            raise ValueError("floor_db should be less tha ceiling_db")

        super().__init__(parent=parent)
        self.setLabel("left", "Frequency", units="Hz")
        self.setLabel("bottom", "Time", units="s")

        color_map = pg.colormap.get(self.settings.color_map)
        self._image = pg.ImageItem(axisOrder="row-major")
        self._image.setColorMap(color_map)
        self._image.setLevels((self.settings.floor_db, self.settings.ceiling_db))
        self.addItem(self._image)

        self._colorbar.setImageItem(self._image, insert_in=self.plotItem)

        self.clear = self._clear_spectrogram

    @staticmethod
    def _validate_result(result:SpectrogramResult) -> None:
        """
        Validate Spectrogram Result

        Parameter
        ---------
        result : SpectrogramResult
            Spectrogram Result to validate
        """
        frequencies = np.asarray(result.frequencies)
        times = np.asarray(result.times)
        magnitudes_db = np.asarray(result.magnitudes_db)

        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError("frequencies must be a non-empty one-dimensional array")
        if times.ndim != 1 or times.size == 0:
            raise ValueError("times must be a non-empty one-dimensional array")
        if magnitudes_db.shape != (frequencies.size, times.size):
            raise ValueError("magnitude_db shape must match (frequencies.size, times.size)")

    @staticmethod
    def _axis_span(values: np.ndarray, fallback: float) -> Tuple[float, float]:
        """
        Return an origin and positive span for an ImageItem axis

        Parameters
        ----------
        values : np.ndarray
            Array of which origin and positive span is to be returned
        fallback : float
            Minimum span for ImageItem axis
        """
        start = values[0]
        if values.size == 1:
            return start, fallback
        return start, max(float(values[-1] - values[0]), fallback)

    def update_spectrogram(self, result: SpectrogramResult) -> None:
        """
        Display the newest rolling STFT history

        Parameter
        ---------
        result : SpectrogramResult
            Spectrogram Result to update
        """
        self._validate_result(result)
        magnitudes_db = np.asarray(result.magnitudes_db)
        times = np.asarray(result.times)
        freqeuncies = np.asarray(result.frequencies)

        self._image.setImage(magnitudes_db, autoLevels=False)
        times_start, times_width = self._axis_span(times, 1e-6)
        freq_start, freq_height = self._axis_span(freqeuncies, 1.0)
        self._image.setRect(QtCore.QRectF(times_start, freq_start, times_width, freq_height))

    def plot_result(self, result:SpectrogramResult) -> None:
        """
        Display a result, using ``update_spectrogram()``

        Parameters
        ----------
        result : SpectrogramResult
            Spectrogram Result to display
        """
        self.update_spectrogram(result)

    def clear_spectrogram(self) -> None:
        """
        Reset displayed image data to the configured dB floor
        """
        image = self._image.image
        if image is None:
            return
        self._image.setImage(np.full(image.shape, fill_value=self.settings.floor_db, dtype=image.dtype), autoLevels=False)
