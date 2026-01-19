"""
Contains functions with different data transforms
"""

from typing import Sequence, Tuple

import numpy as np
import torchvision.transforms as transforms


def get_fundamental_transforms(inp_size: Tuple[int, int]) -> transforms.Compose:
    """Returns the core transforms necessary to feed the images to our model.
    Args:
        inp_size: tuple denoting the dimensions for input to the model

    Returns:
        fundamental_transforms: transforms.compose with the fundamental transforms
    """
    fundamental_transforms = None
    ###########################################################################
    # Student code begins
    ###########################################################################
    fundamental_transforms = transforms.Compose([
        transforms.Resize(inp_size),
        transforms.ToTensor(),
    ])
    return fundamental_transforms

    raise NotImplementedError(
        "`get_fundamental_transforms` function in "
        + "`data_transforms.py` needs to be implemented"
    )

    ###########################################################################
    # Student code ends
    ###########################################################################
    return fundamental_transforms


def get_fundamental_augmentation_transforms(
    inp_size: Tuple[int, int]
) -> transforms.Compose:
    """Returns the data augmentation + core transforms needed to be applied on the train set.
    Suggestions: Jittering, Flipping, Cropping, Rotating.
    Args:
        inp_size: tuple denoting the dimensions for input to the model

    Returns:
        aug_transforms: transforms.compose with all the transforms
    """
    fund_aug_transforms = None
    ###########################################################################
    # Student code begin
    ###########################################################################
    # Use transforms.RandomHorizontalFlip, transforms.ColorJitter
    # Question: How to randomly apply some transforms whilst always applying others?
    
    fund_aug_transforms = transforms.Compose([
        transforms.Resize(inp_size),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.5,  # Range [max(0, 1-0.5), 1+0.5] i.e., [0.5, 1.5]
                contrast=0.5,    # Range [max(0, 1-0.5), 1+0.5] i.e., [0.5, 1.5]
                saturation=0.5,  # Range [max(0, 1-0.5), 1+0.5] i.e., [0.5, 1.5]
                hue=0.2          # Range [-0.2, 0.2])
            )
        #     transforms.RandomResizedCrop(size=inp_size, scale=(0.1,0.9)),
        #     transforms.RandomRotation(10),   # Rotates image by random angle in [-10,10] degrees
        ], p=0.3)  
    ])
    return fund_aug_transforms
    raise NotImplementedError(
        "`get_fundamental_augmentation_transforms` function in "
        + "`data_transforms.py` needs to be implemented"
    )

    ###########################################################################
    # Student code end
    ###########################################################################
    return fund_aug_transforms


def get_fundamental_normalization_transforms(
    inp_size: Tuple[int, int], pixel_mean: Sequence[float], pixel_std: Sequence[float]
) -> transforms.Compose:
    """Returns the core transforms necessary to feed the images to our model alomg with
    normalization.

    Args:
        inp_size: tuple denoting the dimensions for input to the model
        pixel_mean: image channel means, over all images of the raw dataset
        pixel_std: image channel standard deviations, for all images of the raw dataset

    Returns:
        fundamental_transforms: transforms.compose with the fundamental transforms
    """
    fund_norm_transforms = None
    ###########################################################################
    # Student code begins
    ###########################################################################
    fund_norm_transforms = transforms.Compose([
        transforms.Resize(inp_size),
        transforms.ToTensor(),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(pixel_mean, pixel_std), # inplace=False by default should it return a copy?
        # transforms.RandomApply([
        #     transforms.ColorJitter(
        #         brightness=0.5,  # Range [max(0, 1-0.5), 1+0.5] i.e., [0.5, 1.5]
        #         contrast=0.5,    # Range [max(0, 1-0.5), 1+0.5] i.e., [0.5, 1.5]
        #     )
        # #     transforms.RandomResizedCrop(size=inp_size, scale=(0.1,0.9)),
        # #     transforms.RandomRotation(10),   # Rotates image by random angle in [-10,10] degrees
        # ], p=0.5)
        
    ])
    return fund_norm_transforms
    raise NotImplementedError(
        "`get_fundamental_normalization_transforms` function in "
        + "`data_transforms.py` needs to be implemented"
    )

    ###########################################################################
    # Student code ends
    ###########################################################################
    return fund_norm_transforms


def get_all_transforms(
    inp_size: Tuple[int, int], pixel_mean: Sequence[float], pixel_std: Sequence[float]
) -> transforms.Compose:
    """Returns the data augmentation + core transforms needed to be applied on the train set,
    along with normalization. This should just be your previous method + normalization.
    Suggestions: Jittering, Flipping, Cropping, Rotating.
    Args:
        inp_size: tuple denoting the dimensions for input to the model
        pixel_mean: image channel means, over all images of the raw dataset
        pixel_std: image channel standard deviations, for all images of the raw dataset

    Returns:
        aug_transforms: transforms.compose with all the transforms
    """
    all_transforms = None
    ###########################################################################
    # Student code begins
    ###########################################################################
    all_transforms = transforms.Compose([
        transforms.Resize(inp_size),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(pixel_mean, pixel_std), # inplace=False by default should it return a copy?
        transforms.ColorJitter(
            brightness=0.5,  # Range [max(0, 1-0.5), 1+0.5] i.e., [0.5, 1.5]
            contrast=0.5,    # Range [max(0, 1-0.5), 1+0.5] i.e., [0.5, 1.5]
        )
        # transforms.RandomApply([

        #     # transforms.RandomResizedCrop(size=inp_size, scale=(0.1,0.9)),
        #     # transforms.RandomRotation(10),   # Rotates image by random angle in [-10,10] degrees
        # ], p=0.5)
    ])
    return all_transforms
    raise NotImplementedError(
        "`get_all_transforms` function in "
        + "`data_transforms.py` needs to be implemented"
    )

    ###########################################################################
    # Student code ends
    ###########################################################################
    return all_transforms
