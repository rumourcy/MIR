# -*- encoding:utf-8 -*-

import numpy as np

from exceptions import ParameterError


def buf_to_float(x, n_bytes=2, dtype=np.float32):
    """
    type(x) == str
    np.frombuffer 将str转化为int
    """
    scale = 1. / float(1 << ((8 * n_bytes) - 1))
    fmt = '<i{:d}'.format(n_bytes)
    return scale * np.frombuffer(x, fmt).astype(dtype)


def valid_audio(y, mono=True):
    if not isinstance(y, np.ndarray):
        raise ParameterError('data must be of type numpy.ndarray')

    if not np.issubdtype(y.dtype, np.float):
        raise ParameterError('data must be floating-point')

    if mono and y.ndim != 1:
        raise ParameterError('Invalid shape for monophonic audio: '
                             'ndim={:d}, shape={}'.format(y.ndim, y.shape))

    elif y.ndim > 2 or y.ndim == 0:
        raise ParameterError(
            'Audio must have shape (samples,) or (channels, samples). '
            'Received shape={}'.format(y.shape))

    if not np.isfinite(y).all():
        raise ParameterError('Audio buffer is not finite everywhere')

    return True


def fix_length(data, size, axis=-1, **kwargs):
    kwargs.setdefault('mode', 'constant')
    n = data.shape[axis]

    if n > size:
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, size)
        return data[slices]
    elif n < size:
        lengths = [(0, 0)] * data.ndim
        lengths[axis] = (0, size - n)
        return np.pad(data, lengths, **kwargs)

    return data
