"""Live frequency-domain spectrum using PyQtGraph"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg

from src.dsp.spectrum import SpectrumResult

@dataclass(frozen=True)
class SpectrumPlotSettings:
    """
    Configuration settings for a spectrum plot.
    """
    colors: Tuple[str, str]
    sample_rate: float
    line_width: float
    is_scale_log: bool = True
    show_grid: bool = True
    show_peaks: bool = True
    show_peak_labels: bool = False
    show_centroid: bool = False
    show_noise_floor: bool = False

class SpectrumPlot(pg.PlotWidget):
    """
    Display and incrementally update a magnitude spectrum.
    """
    def __init__(
        self,
        settings:SpectrumPlotSettings,
        parent:QtWidgets.QWidget | None = None
    ):

        if settings.sample_rate <= 0:
            raise ValueError("sample_rate must be greater than 0")
        if settings.line_width <= 0:
            raise ValueError("line_width must be greater than 0")

        super().__init__(parent)

        self.sample_rate = float(settings.sample_rate)
        self.line_width = float(settings.line_width)
        self.colors = settings.colors
        self.is_scale_log = bool(settings.is_scale_log)
        self.show_grid = float(settings.show_grid)
        self.show_peaks = float(settings.show_peaks)
        self.show_peak_labels = float(settings.show_peak_labels)
        self.show_centroid = float(settings.show_centroid)
        self.show_noise_floor = settings.show_noise_floor

        self._last_result: SpectrumResult | None = None
        self._peak_labels: list[pg.TextItem] = []

        self.setLabel('left', "Magnitude", units='dB')
        self.setLabel('bottom', "Frequency", units='Hz')
        self.setLogMode(x=self.is_scale_log, y=False)
        left = max(1.0, self.sample_rate / 100000.0) if self.is_scale_log else 0.0
        self.set_frequency_range(left, self.sample_rate / 2.0)
        self.showGrid(x=self.show_grid, y=self.show_grid, alpha=0.3)

        self._line = self.plot(pen=pg.mkPen(self.colors[0], width=self.line_width))
        self._peak_scatter = pg.ScatterPlotItem(
            symbol="x",
            size=10,
            pen=pg.mkPen(self.colors[1], width=2))
        self._centroid_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(self.colors[1], width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self._noise_floor_line = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen(self.colors[1], width=1.5, style=QtCore.Qt.PenStyle.DotLine))
        self.addItem(self._peak_scatter)
        self.addItem(self._centroid_line)
        self.addItem(self._noise_floor_line)
        self._centroid_line.hide()
        self._noise_floor_line.hide()
        
        # PlotWidget assigns its PlotItem.clear proxy to the instance during
        # construction. Restore this component's data-preserving clear method.
        self.clear = self._clear_spectrum

    def _clear_peak_labels(self) -> None:
        """
        Clear labels for peaks
        """
        for label in self._peak_labels:
            self.removeItem(label)
        self._peak_labels.clear()

    def _map_frequencies(self, frequencies: np.ndarray | list) -> np.ndarray:
        """
        Map frequency values to PyQtGraph's logarithmic view coordinates.

        Parameters
        ----------
        frequencies : np.ndarray
            Array of frequencies to map

        Returns
        -------
        np.ndarray
            Logarithmic values of frequencies, if scale is log (``is_scale_log = True``), otherwise normal values of frequencies.
        """
        values = np.asarray(frequencies, dtype=float)
        return np.log10(values) if self.is_scale_log else values

    def _update_overlays(self, result: SpectrumResult) -> None:
        """
        Update peak, centroid and noise floor overlays.

        Parameters
        ----------
        result : SpectrumResult
            New result of spectrum analysis, to update
        """
        self._clear_peak_labels()

        if self.show_peaks:
            self._peak_scatter.setData(
                self._map_frequencies(result.peak_frequencies),
                result.peak_magnitudes_db
            )
        else:
            self._peak_scatter.setData([], [])

        if self.show_peaks and self.show_peak_labels:
            for frequency, magnitude in zip(result.peak_frequencies, result.peak_magnitudes_db):
                label = pg.TextItem(text=f"{frequency:.0f} Hz", color=self.colors[1], anchor=(0.0, 1.0))
                mapped_frequency = self._map_frequencies([frequency])[0]
                label.setPos(float(mapped_frequency), float(magnitude))
                self.addItem(label)
                self._peak_labels.append(label)

        centroid_is_visible = (
            self.show_centroid
            and np.isfinite(result.centroid)
            and (not self.is_scale_log or result.centroid > 0)
        )

        if centroid_is_visible:
            mapped_centroid = self._map_frequencies([result.centroid])[0]
            self._centroid_line.setPos(float(mapped_centroid))
            self._centroid_line.show()
        else:
            self._centroid_line.hide()

        if self.show_noise_floor and np.isfinite(result.noise_floor):
            self._noise_floor_line.setPos(float(result.noise_floor))
            self._noise_floor_line.show()
        else:
            self._noise_floor_line.hide()

    def _clear_spectrum(self) -> None:
        """
        Clear spectrum data and overlays while preserving plot items.
        """
        self._line.clear()
        self._peak_scatter.clear()
        self._clear_peak_labels()
        self._centroid_line.hide()
        self._noise_floor_line.hide()
        self._last_result = None

    def set_frequency_range(self, minimum:float, maximum:float) -> None:
        """
        Set the visible frequency range using values expressed in Hz.

        Parameters
        ----------
        minimum : float
            Left limit of the frequency range
        maximum : float
            Right limit of the frequency range
        """
        if minimum < 0 or maximum <= minimum:
            raise ValueError("frequency range must satisfy 0 <= minimum < maximum")
        if self.is_scale_log and minimum <= 0:
            raise ValueError("minimum frequency must be greater than 0 in log mode")

        limits = self._map_frequencies([minimum, maximum])
        self.setXRange(float(limits[0]), float(limits[1]))

    def update_spectrum(self, result: SpectrumResult) -> None:
        """
        Display spectrum data and all enabled feature overlays.

        Parameters
        ----------
        result : SpectrumResult
            Result of SpectrumAnalyzer to update
        """
        frequencies = np.asarray(result.frequencies)
        magnitudes_db = np.asarray(result.magnitudes_db)

        if frequencies.shape != magnitudes_db.shape:
            raise ValueError("frequencies and magnitudes_db must have matching shapes")

        if self.is_scale_log:
            mask = frequencies > 0
            frequencies = frequencies[mask]
            magnitudes_db = magnitudes_db[mask]

        self._last_result = result
        self._line.setData(frequencies, magnitudes_db)
        self._update_overlays(result)

    def plot_result(self, result:SpectrumResult) -> None:
        """
        Display a result, using ``update_spectrum()``.
        
        Parameters
        ----------
        result : SpectrumResult
            Result of SpectrumAnalyzer to display.
        """
        self.update_spectrum(result)

    def enable_peak_labels(self, choice:bool) -> None:
        """
        Update choice to show peak labels.

        Parameters
        ----------
        choice : bool
        ``True`` or ``False`` value for showing peak labels
        """
        self.show_peak_labels = bool(choice)
        if self._last_result is not None:
            self._update_overlays(self._last_result)

    def enable_centroid(self, choice:bool) -> None:
        """
        Update choice to show centroid.

        Parameters
        ----------
        choice : bool
        ``True`` or ``False`` value for showing centroid
        """
        self.show_centroid = bool(choice)
        if self._last_result is not None:
            self._update_overlays(self._last_result)

    def enable_noise_floor(self, choice:bool) -> None:
        """
        Update choice to show noise floor.

        Parameters
        ----------
        choice : bool
        ``True`` or ``False`` value for noise floor
        """
        self.show_noise_floor = bool(choice)
        if self._last_result is not None:
            self._update_overlays(self._last_result)
