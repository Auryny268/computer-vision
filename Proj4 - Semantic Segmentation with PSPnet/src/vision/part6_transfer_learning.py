import logging
import os
import pdb
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

from src.vision.part5_pspnet import PSPNet
from src.vision.utils import load_class_names, get_imagenet_mean_std, get_logger, normalize_img


_ROOT = Path(__file__).resolve().parent.parent.parent

logger = get_logger()


def load_pretrained_model(args, device: torch.device):
    """Load Pytorch pre-trained PSPNet model from disk of type torch.nn.DataParallel.

    Note that `args.num_model_classes` will be size of logits output.

    Args:
        args:
        device:

    Returns:
        model
    """
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    model = PSPNet(
        layers=args.layers,
        num_classes=args.classes,
        zoom_factor=args.zoom_factor,
        criterion=criterion,
        pretrained=False
    )

    # logger.info(model)
    if device.type == 'cuda':
        cudnn.benchmark = True

    if os.path.isfile(args.model_path):
        logger.info(f"=> loading checkpoint '{args.model_path}'")
        checkpoint = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        logger.info(f"=> loaded checkpoint '{args.model_path}'")
    else:
        raise RuntimeError(f"=> no checkpoint found at '{args.model_path}'")
    model = model.to(device)

    return model



def model_and_optimizer(args, model) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """
    This function is similar to get_model_and_optimizer in Part 3.

    Use the model trained on Camvid as the pretrained PSPNet model, change the
    output classes number to 2 (the number of classes for Kitti).
    Refer to Part 3 for optimizer initialization.

    Args:
        args: object containing specified hyperparameters
        model: pre-trained model on Camvid

    """
    """
    namespace(classes=11, zoom_factor=8, layers=50, ignore_label=255, arch='PSPNet', base_lr=0.001, momentum=0.99, weight_decay=1e-05, pretrained=False
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    # 1. Change number of output classes to 2
    # NOTE: Must manually manipulate model.cls -- apparently trying to call `__create_classifier` will not be successful
    model.cls[4] = nn.Conv2d(512, 2, 1)
    model.aux[4] = nn.Conv2d(256, 2, 1)

    # 2. Initialize optimizer
    param_groups = []
    custom_modules = ["ppm", "cls", "aux"]
    for name, layer in model.named_modules():
        if len(list(layer.children())) == 0:
            lr = args.base_lr
            if any(x in name for x in custom_modules):
                lr = args.base_lr * 10
            param_groups += [{'params': layer.parameters(), 'lr': lr, 'name': name}]
    optimizer = torch.optim.SGD(param_groups, lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # raise NotImplementedError('`model_and_optimizer()` function in ' +
    #     '`part6_transfer_learning.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return model, optimizer
