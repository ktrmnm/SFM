import numpy as np


class Modular:
    def __init__(self, x=None):
        if x is not None:
            self.x = np.ndarray(x, dtype=np.float64)
        else:
            self.x = np.ndarray([], dtype=np.float64)

    @property
    def name(self):
        return "modular"

    @property
    def data(self):
        return { 'x': self.x }
