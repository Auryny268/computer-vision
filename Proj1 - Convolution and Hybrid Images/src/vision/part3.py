#!/usr/bin/python3

"""
PyTorch tutorial on constructing neural networks:
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def my_conv2d_pytorch(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Applies input filter(s) to the input image.

    Args:
        image: Tensor of shape (1, d1, h1, w1)
        kernel: Tensor of shape (N, d1/groups, k, k) to be applied to the image
    Returns:
        filtered_image: Tensor of shape (1, d2, h2, w2) where
           d2 = N
           h2 = (h1 - k + 2 * padding) / stride + 1
           w2 = (w1 - k + 2 * padding) / stride + 1

    HINTS:
    - You should use the 2d convolution operator from torch.nn.functional.
    - In PyTorch, d1 is `in_channels`, and d2 is `out_channels`
    - Make sure to pad the image appropriately (it's a parameter to the
      convolution function you should use here!).
    - You can assume the number of groups is equal to the number of input channels.
    - You can assume only square filters for this function.
    """

    ############################
    ### TODO: YOUR CODE HERE ###
    # kernel: Tensor of shape (N, d1/groups, k, k) to be applied to the image
    # In PyTorch, d1 is `in_channels`, and d2 is `out_channels`
    # You can assume the number of groups is equal to the number of input channels.
    # ^ In other words, the kernel shape is (N, 1, k, k)

    print(f"image shape: {image.shape}")
    print(f"kernel shape: {kernel.shape}")
    d1 = image.shape[1]
    filtered_image = F.conv2d(image, kernel, stride=1, padding='same', groups=d1)
    # filtered_image = F.conv2d(image, kernel, stride=2, padding=1, groups=d1)
    print(f"filtered images shape: {filtered_image.shape}")
    return filtered_image

    # raise NotImplementedError(
    #     "`my_conv2d_pytorch` function in `part3.py` needs to be implemented"
    # )

    # ### END OF STUDENT CODE ####
    # ############################

    # return filtered_image

    """
    Notes from https://www.youtube.com/watch?v=Osx4gfa2e5A:
    torch.nn.conv2d(in_channels=3, out_channels=2, kernel_size=2, stride=1, padding=0)
    
    Suppose the following:
    cin = 3, is num of channels of the input image (ex. RGB)
    cout = 2, is the number of filters we want to apply
    hin = 3, is the height of the input image
    win = 3, is the width of the input image
    hk = 2, is the height of the kernel (i.e. filter)
    wk = 2, is the width of the kernel
    batch(B) = 1, is the number of images we want to process at once 
    padding(p) = 0, is the number of pixels we want to add to each side of the image
    stride(s) = 1, is the number of pixels we want to move the filter each time

    Dimension of the input image = (B, cin, hin, win) = (1, 3, 3, 3)

    Each filter is also composed of cin channels, where the dim(in_channel) = (hk, wk) = (2, 2)
    After convolution, the output image will also have three channels, each with dim(out_channel) = (hout, wout)
    hout = [(hin + 2p - hk) / s] + 1 = [(3 + 0 - 2)) / 1] + 1 = 2]
    wout = [(win + 2p - wk) / s] + 1 = [(3 + 0 - 2)) / 1] + 1 = 2]
    Dimension of the each output image = (Batch, 1, hout, wout)
    total result = [Batch, cout, hout, wout] = (1, 2, 2, 2) 
    """

