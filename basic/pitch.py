# -*- encoding:utf-8 -*-

import numpy as np
import warnings

from basic.specturm import _spectrogram
import basic.time_frequency as time_frequency
import util.utils as utils


def estimate_tuning(y=None, sr=22050, S=None, n_fft=2048,
                    resolution=0.01, bins_per_octave=12, **kawrgs):
    pitch, mag = piptrack(y=y, sr=sr, S=S, n_fft=n_fft, **kawrgs)
    pitch_mask = pitch > 0

    if pitch_mask.any():
        threshold = np.median(mag[pitch_mask])
    else:
        threshold = 0.0

    return pitch_tuning(pitch[(mag >= threshold) & pitch_mask],
                        resolution=resolution, bins_per_octave=bins_per_octave)


def pitch_tuning(frequencies, resolution=0.01, bins_per_octave=12):
    frequencies = np.atleast_1d(frequencies)
    frequencies = frequencies[frequencies > 0]

    if not np.any(frequencies):
        warnings.warn('Trying to estimate tuning from empty frequency set.')
        return 0.0

    residual = np.mod(bins_per_octave *
                      time_frequency.hz_to_octs(frequencies), 1.0)
    residual[residual >= 0.5] -= 1.0

    bins = np.linspace(-0.5, 0.5, int(np.ceil(1./resolution)), endpoint=False)

    counts, tuning = np.histogram(residual, bins)

    return tuning[np.argmax(counts)]


def piptrack(y=None, sr=22050, S=None, n_fft=2048, hop_length=None,
             fmin=150.0, fmax=4000.0, threshold=0.1):
    if hop_length is None:
        hop_length = int(n_fft // 4)

    S, n_fft = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length)

    S = np.abs(S)

    fmin = np.maximum(fmin, 0)
    fmax = np.minimum(fmax, float(sr) / 2)

    fft_freqs = time_frequency.fft_frequencies(sr=sr, n_fft=n_fft)

    avg = 0.5 * (S[2:] - S[:-2])
    shift = 2 * S[1:-1] - S[2:] - S[:-2]
    shift = avg / (shift + (np.abs(shift) < utils.tiny(shift)))

    avg = np.pad(avg, ([1, 1], [0, 0]), mode='constant')
    shift = np.pad(shift, ([1, 1], [0, 0]), mode='constant')

    dskew = 0.5 * avg * shift

    pitches = np.zeros_like(S)
    mags = np.zeros_like(S)

    freq_mask = ((fmin <= fft_freqs) & (fft_freqs < fmax)).reshape((-1, 1))
    idx = np.argwhere(freq_mask &
                      utils.localmax(S * (S > (threshold * S.max(axis=0)))))

    pitches[idx[:, 0], idx[:, 1]] = ((idx[:, 0] + shift[idx[:, 0], idx[:, 1]])
                                     * float(sr) / n_fft)

    mags[idx[:, 0], idx[:, 1]] = (S[idx[:, 0], idx[:, 1]]
                                  + dskew[idx[:, 0], idx[:, 1]])

    return pitches, mags
