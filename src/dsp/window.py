"""Cached window functions for spectral analysis"""
from enum import StrEnum

import numpy as np

class WindowType(StrEnum):
    """
    Supported FFT Window functions
    """
    RECTANGULAR = "rectangular"
    HANN = "hann"
    HAMMING = "hamming"
    BLACKMAN = "blackman"

def generate_window(win_type: WindowType, frame_size: int) -> np.ndarray:
    """
    Generate an immutable ``float32`` window.

    Parameters
    ----------
    win_type : WindowType
        Type of window to generate.
    frame_size : int
        Size of frame for which the window is to be generated.

    Returns
    -------
    np.ndarray
        A ``float32`` window.
    """
    if frame_size <= 0:
        raise ValueError("frame_size must be greater than 0")

    selected = WindowType(win_type)
    match selected:
        case WindowType.RECTANGULAR:
            window =  np.ones(frame_size, dtype=np.float32)
        case WindowType.HANN:
            window = np.hanning(frame_size).astype(np.float32)
        case WindowType.HAMMING:
            window = np.hamming(frame_size).astype(np.float32)
        case WindowType.BLACKMAN:
            window = np.blackman(frame_size).astype(np.float32)

    window.flags.writable = False
    return window

class WindowManager:
    """
    Cache window coefficients by type and frame size.
    """
    def __init__(self):
        self.cache:dict[tuple[WindowType, int], np.ndarray] = {}

    def get_window(self, win_type: WindowType, frame_size: int) -> np.ndarray:
        """
        Return cached read-only window coefficients.

        Parameters
        ----------
        win_type : WindowType
            Type of window to get.
        frame_size : int
            Frame size of the window

        Returns
        -------
        np.ndarray
            Window array.
        """
        if frame_size <= 0:
            raise ValueError("Invalid frame_size")

        selected = WindowType(win_type)

        required_key = (selected, frame_size)
        if required_key not in self.cache:
            self.cache[required_key] = generate_window(selected, frame_size)

        return self.cache[required_key]

    def apply(self, frame: np.ndarray, win_type: WindowType) -> np.ndarray:
        """
        Multiply a 1-D frame by the selected window.

        Parameters
        ----------
        frame : np.ndarray
            Audio frame to apply window function
        win_type : WindowType
            Type of window function to apply to ``frame``

        Returns
        -------
        np.ndarray
            Windowed frame.
        """
        frame_f32 = np.asarray(frame, dtype=np.float32)
        if frame_f32.ndim != 1:
            raise ValueError("Frame must be one-dimensional")
        window = self.get_window(win_type, len(frame_f32))
        return window * frame_f32