import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import signal


class LowPass:
    def __init__(self, cutoff, sample_rate, order):
        self.cutoff = cutoff
        self.sample_rate = sample_rate
        self.order = order

        # self.b, self.a = signal.butter(order, cutoff, btype="lowpass", output="ba", fs=sample_rate)
        self.b = signal.firwin(order + 1, cutoff, pass_zero=True, fs=sample_rate)
        self.a = np.array([1])

    def pad(self, x):
        before = np.repeat(x[..., :1], self.order + 1, axis=-1)
        after = np.repeat(x[..., -1:], self.order + 1, axis=-1)
        return np.concatenate((before, x, after), axis=-1)

    def forward(self, inp):
        x = inp.cpu().detach().numpy() if isinstance(inp, torch.Tensor) else inp
        x = self.pad(x)
        y = signal.lfilter(self.b, self.a, x)
        y = y[..., self.order+1:-(self.order+1)]
        return torch.tensor(y, dtype=inp.dtype, device=inp.device) if isinstance(inp, torch.Tensor) else y

    def __call__(self, inp):
        return self.forward(inp)

    def get_freqz(self, worN=None):
        return signal.freqz(self.b, self.a, worN)

    def plot_freqs(self, worN=None):
        w, h = self.get_freqz(worN=worN)
        f = w / np.pi * self.sample_rate
        plt.plot(f, 20 * np.log10(abs(h)))
        plt.xlabel('Frequency [rad/s]')
        plt.ylabel('Amplitude response [dB]')
        plt.grid(True)
        plt.show()
