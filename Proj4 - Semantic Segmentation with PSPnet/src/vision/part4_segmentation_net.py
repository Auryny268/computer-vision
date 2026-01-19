from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.vision.resnet import resnet50


class SimpleSegmentationNet(nn.Module):
    """
    ResNet backbone, with no increased dilation and no PPM, and a barebones
    classifier.
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 2,
        criterion=nn.CrossEntropyLoss(ignore_index=255),
        deep_base: bool = True,
    ) -> None:
        """ """
        super().__init__()

        self.criterion = criterion
        self.deep_base = deep_base

        resnet = resnet50(pretrained=pretrained, deep_base=True)
        self.resnet = resnet
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.conv2,
            resnet.bn2,
            resnet.relu,
            resnet.conv3,
            resnet.bn3,
            resnet.relu,
            resnet.maxpool,
        )
        # cls = classifier
        self.cls = nn.Conv2d(in_channels=2048, out_channels=num_classes, kernel_size=1)

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass of the network.

        Args:
            x: tensor of shape (N,C,H,W) representing batch of normalized input
                image
            y: tensor of shape (N,H,W) representing batch of ground truth labels

        Returns:
            logits: tensor of shape (N,num_classes,H,W) representing class
                scores at each pixel
            yhat: tensor of shape (N,H,W) representing predicted labels at each
                pixel
            main_loss: loss computed on output of final classifier
            aux_loss: loss computed on output of auxiliary classifier (from
                intermediate output). Note: aux_loss is set to a dummy value,
                since we are not using an auxiliary classifier here, but we
                keep the same API as PSPNet in the next section
        """
        _, _, H, W = x.shape
        # auxilary loss computed after each layer? aux_loss should be container of losses? -> can just return a dummy value here
        # It helps network learn features from intermediate layers
        # Throughout network progresses, downsampling means loss of information
        # aux_loss helps to retain learned features (i.e. train features of earlier layers)
        # Multiple strategies, but typically done before downsampling
        # aux_loss also helps to stabilize gradients during back-propagation

        # GoogLeNet research paper discusses more about this. Also mentioned in the PSPnet paper

        # softmax already done in resnet?

        x = self.layer0(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
       
        # probability of element being class is calculated using softmax -> after we make predictions
         # Softmax-ing generally done in classifier layer

        x = self.cls(x) 
        # Here it's not done anywhere -> so CrossEntropy handles it for us
        # For crossentropy to function, logits must be a prob dist (i.e. softmax)

        main_loss, aux_loss = None, None

        ########################################################################
        # TODO: YOUR CODE HERE                                                 #
        # Upsample the output to (H,W) using Pytorch's functional              #
        # `interpolate`, then compute the loss, and the predicted label per    #
        # pixel (yhat).                                                        #
        ########################################################################
        # print(f"output shape: {x.shape}")
        # After model pass, results are shape 5x11x7x7 -> 5 batches, 11 channels each with 7x7 "image"
        # F.interpolate automatically resizes height/width instead of batch_size/num_channels if given two-ple (haha)
        logits = F.interpolate(x, size=(H, W))
        if y != None:
            main_loss  = self.criterion(logits, y)
            aux_loss = torch.Tensor([0])
        yhat = torch.argmax(logits, dim=1)

        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################
        return logits, yhat, main_loss, aux_loss
