# -*- encoding:utf-8 -*-

import numpy as np

from basic.specturm import _spectrogram
from basic.pitch import estimate_tuning
import util.utils as utils
import filters


def chroma_stft(y=None, sr=22050, S=None, norm=np.inf, n_fft=2048,
                hop_length=512, tuning=None, **kwargs):
    S, n_fft = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length,
                            power=2)

    n_chroma = kwargs.get('n_chroma', 12)

    if tuning is None:
        tuning = estimate_tuning(S=S, sr=sr, bins_per_octave=n_chroma)

    if 'A440' not in kwargs:
        kwargs['A440'] = 440.0 * 2.0**(float(tuning) / n_chroma)

    chromafb = filters.chroma(sr, n_fft, **kwargs)
    raw_chroma = np.dot(chromafb, S)

    return utils.normalize(raw_chroma, norm=norm, axis=0)
