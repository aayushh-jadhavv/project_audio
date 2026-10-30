"""Reusable PyQt6 transport and DSP control widgets."""

from PyQt6 import QtCore, QtWidgets

from src.dsp.equalizer import EqualizerBand

class TransportControls(QtWidgets.QGroupBox):
    """
    Start, pause and stop controls.

    Parameters
    ----------
    parent : QtWidgets.QWidget | None, default=None
        ``QWidget`` instance which should own the instance of this class. Default is None, so no object owns the instance.
    """

    start_requested = QtCore.pyqtSignal()
    pause_requested = QtCore.pyqtSignal()
    stop_requested = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__("Controls", parent)
        layout = QtWidgets.QHBoxLayout(self)

        self.start_button = QtWidgets.QPushButton("Start")
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.stop_button = QtWidgets.QPushButton("Stop")

        self.start_button.clicked.connect(self.start_requested.emit)
        self.pause_button.clicked.connect(self.pause_requested.emit)
        self.stop_button.clicked.connect(self.stop_requested.emit)

        layout.addWidget(self.start_button)
        layout.addWidget(self.pause_button)
        layout.addWidget(self.stop_button)

class EqualizerControls(QtWidgets.QGroupBox):
    """
    Vertical gain controls for each parametric EQ band.

    Parameters
    ----------
    bands : tuple[EqualizerBand, ...]
        Tuple of Parametric Equalizer bands
    parent : QtWidgets.QWidget | None, default=None
        ``QWidget`` instance which should own the instance of this class. Default is None, so no object owns the instance.
    """

    band_gain_changed = QtCore.pyqtSignal(int, float)

    def __init__(self, bands:tuple[EqualizerBand, ...], parent:QtWidgets.QWidget | None = None):
        super().__init__("Parametric Equalizer", parent)

        self._sliders: list[QtWidgets.QSlider] = []
        self._value_labels: list[QtWidgets.QLabel] = []
        layout = QtWidgets.QHBoxLayout(self)

        for i, band in enumerate(bands):
            column = QtWidgets.QVBoxLayout()
            gain_label = QtWidgets.QLabel(f"{band.gain_db:+.1f} dB")
            gain_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
            slider.setRange(-120, 120)
            slider.setValue(round(band.gain_db * 10))
            slider.setTickInterval(30)
            slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBothSides)
            slider.valueChanged.connect(lambda value, index=i: self._emit_gain(index, value))

            frequency_label = QtWidgets.QLabel(self._format_frequency(band.center_frequency))
            frequency_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            
            column.addWidget(gain_label)
            column.addWidget(slider, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
            column.addWidget(frequency_label)

            layout.addLayout(column)

            self._sliders.append(slider)
            self._value_labels.append(gain_label)

    @staticmethod
    def _format_frequency(frequency: float) -> str:
        if frequency >= 1000:
            return f"{frequency / 1000:g} kHz"
        return f"{frequency:g} Hz"

    def _emit_gain(self, index: int, value: int) -> None:
        gain_db = value / 10.0
        self._value_labels[index].setText(f"{gain_db:+.1f} dB")
        self.band_gain_changed.emit(index, gain_db)