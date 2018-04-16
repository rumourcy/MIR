# -*- encoding:utf-8 -*-

import numpy as np


def fft_frequencies(sr=22050, n_fft=2048):
    return np.linspace(0, float(sr) / 2, int(1 + n_fft/2), endpoint=True)


def hz_to_octs(frequencies, A440=440.0):
    return np.log2(np.asanyarray(frequencies) / (float(A440) / 16))
