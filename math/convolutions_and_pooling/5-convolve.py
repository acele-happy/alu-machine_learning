#!/usr/bin/env python3
"""Convolution on images using multiple kernels."""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels.

    Args:
        images: numpy.ndarray of shape (m, h, w, c)
        kernels: numpy.ndarray of shape (kh, kw, c, nc)
        padding: tuple of (ph, pw), 'same', or 'valid'
        stride: tuple of (sh, sw)

    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = kh // 2
        pw = kw // 2
    else:
        ph, pw = padding

    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    if padding == 'same':
        new_h = int(np.ceil(h / sh))
        new_w = int(np.ceil(w / sw))
    else:
        new_h = (h + 2 * ph - kh) // sh + 1
        new_w = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, new_h, new_w, nc))

    for i in range(new_h):
        for j in range(new_w):
            for k in range(nc):
                output[:, i, j, k] = np.sum(
                    padded[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :] *
                    kernels[:, :, :, k],
                    axis=(1, 2, 3)
                )

    return output
