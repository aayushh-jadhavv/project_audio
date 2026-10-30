"""Live waveform visualizer using PyQtGraph"""
from typing import Tuple
from dataclasses import dataclass

import numpy as np
from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg

@dataclass(frozen=True)
class WaveformSettings:
    """
    Settings for a waveform plot
    """
    sample_rate: float
    color: Tuple[str, str]
    line_width: float

class WaveformPlot(pg.PlotWidget):
    """
    Live time-domain waveform plot in PyQtGraph

    Parameters
    ----------
    settings : WaveformSettings
        Settings for the plot
    """
    def __init__(self, settings : WaveformSettings, parent : QtWidgets.QWidget | None = None):
        super().__init__(parent=parent)

        if settings.sample_rate <= 0:
            raise ValueError("sample_rate must be greater than 0")

        self.sr = float(settings.sample_rate)

        self.setLabel("Left", "Amplitude")
        self.setLabel("bottom", "Time", units="s")
        self.setYRange(-1.0, 1.0)
        self.showGrid(x=True, y=True, alpha=0.2)

        self._curve = self.plot(pen=pg.mkPen(color=settings.color, width=settings.line_width))
        self._times = np.empty(0)

    def update_samples(self, samples : np.ndarray) -> None:
        """
        Update plot

        Parameters
        ----------
        samples : np.ndarray
            Array of samples
        """
        values = np.asarray(samples, dtype=np.float32)

        if values.ndim != 1:
            raise ValueError("samples must be one-dimensional")
        if values.size != self._times.size:
            self._times = np.arange(values.size) / self.sr

        self._curve.setData(self._times, values)

    def clear(self) -> None:
        """
        Clear the plot
        """
        self._curve.clear()