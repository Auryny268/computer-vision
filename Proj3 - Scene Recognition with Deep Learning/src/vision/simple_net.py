import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        """
        Init function to define the layers and loss function

        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means

        """
        super(SimpleNet, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, 5),
            nn.MaxPool2d(3,3),
            nn.ReLU(),
            nn.Conv2d(10, 20, 5),
            nn.MaxPool2d(3,3),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            # Draws weights using a uniform distribution
            nn.Linear(500,100),
            nn.Linear(100,15)
        )
        # Use negative log-likelihood (good for classification w/ small number of classes)
        # (Double-check) nn.NLLLoss assigns class whereas nn.NLLLoss assigns instance
        # Expectation is to play around with different loss criterions
        self.loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        # self.loss_criterion =nn.NLLLoss()

        ############################################################################
        # Student code begin
        ############################################################################

        # raise NotImplementedError(
        #     "`__init__` function in "
        #     + "`simple_net.py` needs to be implemented"
        # )

        ############################################################################
        # Student code end
        ############################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass with the net

        Args:
        -   x: the input image [Dim: (N,C,H,W)]
        Returns:
        -   y: the output (raw scores) of the net [Dim: (N,15)]
        """
        model_output = None
        ############################################################################
        # Student code begin
        ############################################################################
        # TA solution: model_output = self.fc_layers((self.conv_layers(x)).view(x.shape[0],-1))
        conv_output = self.conv_layers(x)
        # Starts flattening at dim=1 instead of dim=0
        # AKA flattens each batch
        flattened = torch.flatten(conv_output, start_dim=1)
        # model_output = self.fc_layers(flattened)
        logits = self.fc_layers(flattened)
        # Applies softmax then log along dim=1 (For each batch) -> need to figure out why
        model_output = torch.log_softmax(logits, dim=1) 
        # print(model_output.shape)
        # print(model_output[0])
        return model_output
        raise NotImplementedError(
            "`forward` function in "
            + "`simple_net.py` needs to be implemented"
        )

        ############################################################################
        # Student code end
        ############################################################################

        return model_output
