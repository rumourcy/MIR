# -*- encoding:utf-8 -*-

import numpy as np
import scipy.fftpack as fft
import scipy.signal

import util.utils as utils


def stft(y, n_fft=2048, hop_length=None, win_length=None, window='hann',
         center=True, dtype=np.complex64, pad_mode='reflect'):
    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = int(win_length // 4)

    fft_window = scipy.signal.get_window(window, win_length, fftbins=True)
    fft_window = utils.pad_center(fft_window, n_fft)
    fft_window = fft_window.reshape((-1, 1))

    utils.valid_audio(y)

    if center:
        y = np.pad(y, int(n_fft // 2), mode=pad_mode)

    y_frames = utils.frame(y, frame_length=n_fft, hop_length=hop_length)
    stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
                           dtype=dtype, order='F')

    n_columns = int(utils.MAX_MEM_BLOCK / (
        stft_matrix.shape[0] * stft_matrix.itemsize))

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])
        stft_matrix[:, bl_s:bl_t] = fft.fft(
            fft_window * y_frames[:, bl_s:bl_t],
            axis=0)[:stft_matrix.shape[0]]

    return stft_matrix


def _spectrogram(y=None, S=None, n_fft=2048, hop_length=512, power=1):
    if S is not None:
        n_fft = 2 * (S.shape[0] - 1)
    else:
        S = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length))**power

    return S, n_fft
