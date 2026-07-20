#!/usr/bin/env python3
"""Performs a valid convolution on grayscale images."""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    images: numpy.ndarray of shape (m, h, w)
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    kernel: numpy.ndarray of shape (kh, kw)
        kh is the height of the kernel
        kw is the width of the kernel

    Returns: numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    output_h = h - kh + 1
    output_w = w - kw + 1

    convolved = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            image_slice = images[:, i:i + kh, j:j + kw]
            convolved[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return convolved