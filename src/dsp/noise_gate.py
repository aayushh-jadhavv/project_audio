"""Envelope-based noise gate for suppressing low-level background noise."""
from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class NoiseGateSettings:
    """Threshold and smoothing values for :class:`NoiseGate`."""
    threshold_db: float = -50.0
    attack_ms: float = 5.0
    release_ms: float = 100.0

class NoiseGate:
    """Apply a smoothed binary gain based on the input envelope."""
    def __init__(
            self,
            sample_rate: float,
            settings: NoiseGateSettings | None = None,
            enabled: bool = False
    ):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be greater than 0")

        self.sr = float(sample_rate)
        self.settings = settings or NoiseGateSettings()
        if self.settings.attack_ms <= 0 or self.settings.release_ms <= 0:
            raise ValueError("attack_ms and release_ms must be greater than 0")

        self.enabled = enabled
        self._gain = 0.0
        self._attack_coefficients = self._time_coefficients(self.settings.attack_ms)
        self._release_coefficients = self._time_coefficients(self.settings.release_ms)

    def _time_coefficients(self, milliseconds: float) -> float:
        """
        Return coefficients for attack or release.

        Parameters
        ----------
        milliseconds : float
            time interval of attack/release

        Returns
        -------
        float
            Coefficients for gain-smoothing
        """
        seconds = milliseconds / 1000.0
        return float(np.exp(-1.0 / (seconds * self.sr)))

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Gate one consecutive mono frame.

        Parameters
        ----------
        frame : np.ndarray
            mono frame to gate

        Returns
        -------
        np.ndarray
            Array containing processed samples of the frame
        """
        frame_f32 = np.asarray(frame, dtype=np.float32)
        if frame_f32.ndim != 1:
            raise ValueError("frame must be one-dimensional")
        if not self.enabled:
            return frame_f32.copy()

        threshold = 10.0 ** (self.settings.threshold_db / 20.0)
        output = np.empty_like(frame_f32)
        for index, sample in enumerate(frame_f32):
            target = 1.0 if abs(float(sample)) >= threshold else 0.0
            coefficient = self._attack_coefficients if target > self._gain else self._release_coefficients
            gain = coefficient * self._gain + (1.0 - coefficient) * target
            output[index] = sample * self._gain

        return output

    def reset(self) -> None:
        """
        Close the gate and clear smoothing state.
        """
        self._gain = 0.0