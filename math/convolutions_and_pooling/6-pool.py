#!/usr/bin/env python3
"""Pooling on images."""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    Args:
        images: numpy.ndarray of shape (m, h, w, c)
        kernel_shape: tuple of (kh, kw)
        stride: tuple of (sh, sw)
        mode: 'max' or 'avg'

    Returns:
        numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    new_h = (h - kh) // sh + 1
    new_w = (w - kw) // sw + 1
    output = np.zeros((m, new_h, new_w, c))

    for i in range(new_h):
        for j in range(new_w):
            region = images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            if mode == 'max':
                output[:, i, j, :] = np.max(region, axis=(1, 2))
            else:
                output[:, i, j, :] = np.mean(region, axis=(1, 2))

    return output
