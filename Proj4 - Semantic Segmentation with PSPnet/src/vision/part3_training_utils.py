from typing import Tuple

import torch
from torch import nn

import src.vision.cv2_transforms as transform
from src.vision.part5_pspnet import PSPNet
from src.vision.part4_segmentation_net import SimpleSegmentationNet


def get_model_and_optimizer(args) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """
    Create your model, optimizer and configure the initial learning rates.

    Use the SGD optimizer, use a parameters list, and set the momentum and
    weight decay for each parameter group according to the parameter values
    in `args`.

    Create 5 param groups for the 0th + 1st,2nd,3rd,4th ResNet layer modules,
    and then add separate groups afterwards for the classifier and/or PPM
    heads.

    You should set the learning rate for the resnet layers to the base learning
    rate (args.base_lr), and you should set the learning rate for the new
    PSPNet PPM and classifiers to be 10 times the base learning rate.

    Args:
        args: object containing specified hyperparameters, including the "arch"
           parameter that determines whether we should return PSPNet or the
           SimpleSegmentationNet
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    """
    Args:
    - classes=11
    - zoom_factor=8
    - layers=50
    - ignore_label=255,
    - arch='SimpleSegmentationNet'
    - base_lr=0.001
    - momentum=0.99
    - weight_decay=1e-5
    - pretrained=False
    """
    print(args)

    """
    namespace(classes=11, zoom_factor=8, layers=50, ignore_label=255, arch='PSPNet', base_lr=0.001, momentum=0.99, weight_decay=1e-05, pretrained=False, use_ppm=True)
    """
    # 1. Docstring is wrong -> make model first then 
    if args.arch == 'PSPNet':
        model = PSPNet(args.layers, bins=(1,2,3,6), num_classes=args.classes, zoom_factor=args.zoom_factor, pretrained=args.pretrained, use_ppm=args.use_ppm)
    else:
        model = SimpleSegmentationNet(args.pretrained, args.classes)
    param_groups = []
    # 2. Create params groups manually
    custom_modules = ["ppm", "cls", "aux"]
    for name, layer in model.named_modules():
        # Need to check for "leaf" layers using .children()
        # print(i, name)
        # print(len(list(layer.children())) == 0)
        if len(list(layer.children())) == 0:
            lr = args.base_lr*10 if any(x in name for x in custom_modules) else args.base_lr
            param_groups += [{'params': layer.parameters(), 'lr': lr, 'name': name}]
            # if any(x in name for x in custom_modules):
            #     param_groups += [{'params': layer.parameters(), 'lr': args.base_lr*10, 'name': name}]
            # else:
            #     param_groups += [{'params': layer.parameters(), 'lr': args.base_lr, 'name': name}]
    # Need to send in iterable of dictionaries for param_groups
    optimizer = torch.optim.SGD(param_groups, lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    return model, optimizer

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


def update_learning_rate(current_lr: float, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
    """
    Given an updated current learning rate, set the ResNet modules to this
    current learning rate, and the classifiers/PPM module to 10x the current
    lr.

    Hint: You can loop over the dictionaries in the optimizer.param_groups
    list, and set a new "lr" entry for each one. They will be in the same order
    you added them above, so if the first N modules should have low learning
    rate, and the next M modules should have a higher learning rate, this
    should be easy modify in two loops.

    Note: this depends upon how you implemented the param groups above.
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    custom_modules = ["ppm", "cls", "aux"]
    for param_group in optimizer.param_groups:
        # Need to check for "leaf" layers using .children()
        # print(i, name)
        # print(len(list(layer.children())) == 0)
        new_lr = current_lr
        if any(x in param_group['name'] for x in custom_modules):
            new_lr = current_lr * 10
        param_group['lr'] = new_lr

    # raise NotImplementedError('`update_learning_rate()` function in ' +
    #     '`part3_training_utils.py` needs to be implemented')


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return optimizer


def get_train_transform(args) -> transform.Compose:
    """
    Compose together with transform.Compose() a series of data proprocessing
    transformations for the training split, with data augmentation. Use the classes
    from `src/vision/cv2_transforms`, imported above as `transform`.

    These should include resizing the short side of the image to args.short_size,
    then random horizontal flipping, blurring, rotation, scaling (in any order),
    followed by taking a random crop of size (args.train_h, args.train_w), converting
    the Numpy array to a Pytorch tensor, and then normalizing by the
    Imagenet mean and std (provided here).

    Note that your scaling should be confined to the [scale_min,scale_max] params in the
    args. Also, your rotation should be confined to the [rotate_min,rotate_max] params.

    To prevent black artifacts after a rotation or a random crop, specify the paddings
    to be equal to the Imagenet mean to pad any black regions.

    You should set such artifact regions of the ground truth to be ignored.

    Use the classes
    from `src/vision/cv2_transforms`, imported above as `transform`.

    Args:
        args: object containing specified hyperparameters

    Returns:
        train_transform
    """

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    train_transform = transform.Compose([
         # 1. Resize the short side of the image to args.short_size (ResizeShort)
        transform.ResizeShort(size=args.short_size),
        # 2. Do random horizontal flipping, blurring, rotation, scaling (in any order)
        transform.RandomHorizontalFlip(p=0.5),
        transform.RandomGaussianBlur(), # radius is 5 by default
        transform.RandScale(scale=(args.scale_min, args.scale_max)),
        transform.RandRotate(rotate=(args.rotate_min, args.rotate_max), padding=mean, ignore_label=args.ignore_label, p=0.5),
        # 3. Take random crop of size (args.train_h, args.train_w) with padding=mean(Imagenet)
        transform.Crop(size=(args.train_h, args.train_w), crop_type='rand', padding=mean, ignore_label=args.ignore_label),
        # 4. Convert np array to torch tensor
        transform.ToTensor(),
        # 5. Normalize by Imagenet mean/std
        transform.Normalize(mean=mean, std=std)
    ])

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return train_transform


def get_val_transform(args) -> transform.Compose:
    """
    Compose together with transform.Compose() a series of data proprocessing
    transformations for the val split, with no data augmentation. Use the classes
    from `src/vision/cv2_transforms`, imported above as `transform`.

    These should include resizing the short side of the image to args.short_size,
    taking a *center* crop of size (args.train_h, args.train_w) with a padding equal
    to the Imagenet mean, converting the Numpy array to a Pytorch tensor, and then
    normalizing by the Imagenet mean and std (provided here).

    Args:
        args: object containing specified hyperparameters

    Returns:
        val_transform
    """

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    # Why the fuck did we reinvent the wheel and write our own transforms instead of using torch's transforms???
    # Torch compose takes list as param!
    val_transform = transform.Compose([
         # 1. Resize the short side of the image to args.short_size (ResizeShort)
        transform.ResizeShort(size=args.short_size),
        # 2. Take center crop of size (args.train_h, args.train_w) with padding=mean(Imagenet)
        # Do we have to pass args.ignore_label here??
        transform.Crop(size=(args.train_h, args.train_w), crop_type='center', padding=mean),
        # 3. Convert np array to torch tensor
        transform.ToTensor(),
        # 4. Normalize by Imagenet mean/std
        transform.Normalize(mean=mean, std=std)
    ])

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return val_transform
