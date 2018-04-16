# -*- encoding:utf-8 -*-

import os

import numpy as np
import audioread
import scipy.signal

import util.utils as utils


def load(path, sr=22050, mono=True, offset=0.0, duration=None, dtype=np.float32):
    y = []
    with audioread.audio_open(os.path.realpath(path)) as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels
        # np.round 四舍五入
        s_start = int(np.round(sr_native * offset)) * n_channels

        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + (int(
                np.round(sr_native * duration) * n_channels))

        n = 0

        for frame in input_file:
            frame = utils.buf_to_float(frame, dtype=dtype)
            # 当前开始位置
            n_prev = n
            # 当前结束位置
            n = n + len(frame)

            if n < s_start:
                continue
            if s_end < n_prev:
                break
            if s_end < n:
                frame = frame[:s_end - n_prev]
            if n_prev <= s_start <= n:
                frame = frame[(s_start - n_prev):]
            y.append(frame)

    if y:
        y = np.concatenate(y)

        if n_channels > 1:
            y = y.reshape((-1, n_channels)).T
            if mono:
                y = to_mono(y)

        if sr is not None:
            y = resample(y, sr_native, sr)
        else:
            sr = sr_native

    y = np.ascontiguousarray(y, dtype=dtype)

    return (y, sr)


def to_mono(y):
    utils.valid_audio(y, mono=False)

    if y.ndim > 1:
        # 求均值
        y = np.mean(y, axis=0)

    return y


def resample(y, orig_sr, target_sr, fix=True, scale=False, **kwargs):
    utils.valid_audio(y, mono=False)

    if orig_sr == target_sr:
        return y

    ratio = float(target_sr) / orig_sr
    n_samples = int(np.ceil(y.shape[-1] * ratio))
    y_hat = scipy.signal.resample(y, n_samples, axis=-1)

    if fix:
        y_hat = utils.fix_length(y_hat, n_samples, **kwargs)

    if scale:
        y_hat /= np.sqrt(ratio)

    return np.ascontiguousarray(y_hat, dtype=y.dtype)
