# -*- encoding:utf-8 -*-

import numpy as np
from numpy.lib.stride_tricks import as_strided

from exceptions import ParameterError

MAX_MEM_BLOCK = 2**8 * 2**10


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


def frame(y, frame_length=2048, hop_length=512):
    if not isinstance(y, np.ndarray):
        raise ParameterError('Input must be of type numpy.ndarray, '
                             'given type(y)={}'.format(type(y)))

    if y.ndim != 1:
        raise ParameterError('Input must be one-dimensional, '
                             'given y.ndim={}'.format(y.ndim))

    if len(y) < frame_length:
        raise ParameterError('Buffer is too short (n={:d})'
                             ' for frame_length={:d}'.format(
                                 len(y), frame_length))

    if hop_length < 1:
        raise ParameterError('Invalid hop_length: {:d}'.format(hop_length))

    if not y.flags['C_CONTIGUOUS']:
        raise ParameterError('Input buffer must be contiguous.')

    n_frames = 1 + int((len(y) - frame_length) / hop_length)
    y_frames = as_strided(y, shape=(frame_length, n_frames),
                          strides=(y.itemsize, hop_length * y.itemsize))
    return y_frames


def pad_center(data, size, axis=-1, **kwargs):
    kwargs.setdefault('mode', 'constant')
    n = data.shape[axis]
    lpad = int((size-n) // 2)
    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise ParameterError(('Target size ({:d}) must be '
                              'at least input size ({:d})').format(size, n))

    return np.pad(data, lengths, **kwargs)


def normalize(S, norm=np.inf, axis=0, threshold=None, fill=None):
    if threshold is None:
        threshold = tiny(S)
    elif threshold <= 0:
        raise ParameterError('threshold={} must be strictly '
                             'positive'.format(threshold))

    if fill not in [None, False, True]:
        raise ParameterError('fill={} must be None or boolean'.format(fill))

    if not np.all(np.isfinite(S)):
        raise ParameterError('Input must be finite')

    mag = np.abs(S).astype(np.float)

    fill_norm = 1

    if norm == np.inf:
        length = np.max(mag, axis=axis, keepdims=True)
    elif norm == -np.inf:
        length = np.min(mag, axis=axis, keepdims=True)
    elif norm == 0:
        if fill is True:
            raise ParameterError('Cannot normalize with norm=0 and fill=True')
        length = np.sum(mag > 0, axis=axis, keepdims=True, dtype=mag.dtype)
    elif np.issubdtype(type(norm), np.number) and norm > 0:
        length = np.sum(mag**norm, axis=axis, keepdims=True)**(1./norm)
        if axis is None:
            fill_norm = mag.size**(-1./norm)
        else:
            fill_norm = mag.shape[axis]**(-1./norm)
    elif norm is None:
        return S
    else:
        raise ParameterError('Unsupported norm: {}'.format(repr(norm)))

    small_idx = length < threshold

    Snorm = np.empty_like(S)
    if fill is None:
        length[small_idx] = 1.0
        Snorm[:] = S / length
    elif fill:
        length[small_idx] = np.nan
        Snorm[:] = S / length
        Snorm[np.isnan(Snorm)] = fill_norm
    else:
        length[small_idx] = np.inf
        Snorm[:] = S / length

    return Snorm


def localmax(x, axis=0):
    paddings = [(0, 0)] * x.ndim
    paddings[axis] = (1, 1)

    x_pad = np.pad(x, paddings, mode='edge')

    inds1 = [slice(None)] * x.ndim
    inds1[axis] = slice(0, -2)

    inds2 = [slice(None)] * x.ndim
    inds2[axis] = slice(2, x_pad.shape[axis])

    return (x > x_pad[inds1]) & (x >= x_pad[inds2])


def tiny(x):
    x = np.asarray(x)
    if np.issubdtype(x.dtype, float) or np.issubdtype(x.dtype, complex):
        dtype = x.dtype
    else:
        dtype = np.float32

    return np.finfo(dtype).tiny
