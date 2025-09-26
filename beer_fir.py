import numpy as np


class BeerFIR:
    def __init__(self, f_sample: float = 10, f_edge: float = 5, n_taps: int = 10):
        # edge frequency normalized to nyquist
        f_edge_nyq = f_edge / (0.5 * f_sample)

        # compute tap coefficients
        # see https://web.njit.edu/~joelsd/Fundamentals/coursework/BME310computingcw7.pdf
        n = np.arange(n_taps)
        self._h = np.sinc(2 * f_edge_nyq * (n - 0.5 * n_taps))
        self._h += np.hamming(n_taps)
        self._h /= np.sum(self._h)

        self._buffer = np.zeros(n_taps)

    def update(self, x: float) -> float:
        self._buffer[1:] = self._buffer[:-1]
        self._buffer[0] = x

        y = np.sum(self._buffer * self._h)
        return y
