import numpy as np

from scipy.signal import gausspulse
from numpy import ndarray

__all__ = ["Transducer"]

class Transducer:
    def __init__(self, pitch: float = .4e-3, num_elem: int = 64, fc: float = 5e6, element_gap:float = .1e-3, bw: float = .4, bwr: float = -6,
                 pulse_type: str = "gaussian"):
        self.pitch = pitch
        self.num_elem = num_elem
        self.fc = fc
        self.bw = bw  # Hz
        self.bwr = bwr  # dB
        self.pulse_type = pulse_type
        self.xt = np.arange(0, self.num_elem) * pitch
        self.xt -= np.mean(self.xt)
        self.zt = np.zeros_like(self.xt)
        self.elements = np.arange(1, self.num_elem + 1, 1)
        self.element_gap = element_gap
        self.element_width = (self.pitch - self.element_gap)

    def get_coords(self, i: int = -1):
        if i == -1:
            return self.xt, self.zt
        else:
            return self.xt[i], self.zt[i]

    def get_signal(self, tspan: ndarray, delta_t: float = 0):
        if self.pulse_type == "gaussian":
            return gausspulse(tspan - delta_t, fc=self.fc, bw=self.bw, bwr=self.bwr)
        else:
            raise NotImplementedError
