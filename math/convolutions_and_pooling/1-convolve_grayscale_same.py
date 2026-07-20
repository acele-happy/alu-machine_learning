#!/usr/bin/env python3
"""Performs a same convolution on grayscale images."""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    images: numpy.ndarray of shape (m, h, w)
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    kernel: numpy.ndarray of shape (kh, kw)
        kh is the height of the kernel
        kw is the width of the kernel

    If necessary, the image is padded with 0's.

    Returns: numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    pad_h_top = (kh - 1) // 2
    pad_h_bottom = kh // 2
    pad_w_left = (kw - 1) // 2
    pad_w_right = kw // 2

    padded_images = np.pad(
        images,
        pad_width=((0, 0), (pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right)),
        mode='constant',
        constant_values=0
    )

    convolved = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            image_slice = padded_images[:, i:i + kh, j:j + kw]
            convolved[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return convolved
