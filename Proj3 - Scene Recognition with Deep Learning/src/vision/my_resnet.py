import torch
import torch.nn as nn
from torchvision.models import resnet18


class MyResNet18(nn.Module):
    def __init__(self):
        """Initialize network layers.

        Note: Do not forget to freeze the layers of ResNet except the last one
        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means

        Download pretrained resnet using pytorch's API (Hint: see the import statements)
        """
        super().__init__()

        self.conv_layers = None
        self.fc_layers = None
        self.loss_criterion = None


        # Step 1: Copy network architecture and weight of all but the last fc_layers
        model_ft = resnet18(weights='IMAGENET1K_V1')
        # print(*list(model_ft.children()))
        self.conv_layers = nn.Sequential(*list(model_ft.children())[:-1])
        
        # print(*list(model_ft.children())[-1])

        # Step 2: Freeze conv_layers
        for param in self.conv_layers.parameters():
            param.requires_grad = False
        
        # Step 3: Pre-Freeze first two FC layers
        # frozen_fc = nn.Sequential(
        #     nn.Linear(in_features=16384, out_features=512),
        #     nn.Linear(in_features=512, out_features=100)
        # )
        # for param in frozen_fc.parameters():
        #     param.requires_grad = False
        
        # Step 4: Add fc_layers
        self.fc_layers = nn.Sequential(
            # frozen_fc,
            nn.Linear(in_features=512, out_features=15)
            # nn.Linear(16384, 15)
        )

        # Step 5: Define Loss Criterion
        self.loss_criterion = nn.CrossEntropyLoss(reduction="mean")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass with the net, duplicating grayscale channel to 3-channel.

        Args:
            x: tensor of shape (N,C,H,W) representing input batch of images
        Returns:
            y: tensor of shape (N,num_classes) representing the output (raw scores) of the net
                Note: we set num_classes=15
        """
        model_output = None
        x = x.repeat(1, 3, 1, 1)  # as ResNet accepts 3-channel color images
        conv_output = self.conv_layers(x)
        flattened = torch.flatten(conv_output, start_dim=1)
        model_output = self.fc_layers(flattened)
        return model_output
