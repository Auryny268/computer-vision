import torch
import torch.nn as nn

# Use option=up/down to swap lines

"""
TODO (2.3): The dropout layer has only one free parameter (the dropout rate), which is the proportion of connections
that are randomly deleted. The default of 0.5 should be fine. Insert a dropout layer between your convolu-
tional layers in the SimpleNetFinal class in simple_net_final.py. In particular, insert it directly before your
last convolutional layer. Your validation accuracy should increase by another 10%. Your train loss should
4
decrease much more slowly. That’s to be expected–you’re making life much harder for the training algorithm
by cutting out connections randomly.

TODO (2.4): Let’s make our network deeper by adding an additional convolutional layer to simple_net_final.py. In
fact, we probably don’t want to add just a convolutional layer, but another max-pooling layer and ReLU
layer as well.  For example, you might insert a convolutional layer after the existing ReLU layer with a 5 × 5
spatial support, followed by a max-pool over a 3× 3 window with a stride of 2. You can reduce the max-pool
window in the previous layer, adjust padding, and reduce the spatial resolution of the final layer until your
network’s final layer (not counting the softmax) has a data size of 15. You also need to make sure that the
5
data depth output by any channel matches the data depth input to the following channel. For instance,
maybe your new convolutional layer takes in the 10 channels of the first layer but outputs 15 channels. The
final layer would then need to have its weights initialized accordingly to account for the fact that is operates
on a 15-channel image instead of a 10-channel image


TODO (2.5): In particular, let’s add a batch normalization layer after each convolutional layer
(except for the last) in simple_net_final.py. If you have 4 total convolutional layers we will add 3 batch
normalization layers. You can check out nn.BatchNorm2d(num_features=...). You will also need to initialize
the weights of the batch normalization layers. Set weight to 1 and bias to 0.
"""

# NOTES:
# nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, 
#                                                           padding_mode='zeros', device=None, dtype=None)
# nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
# nn.ReLU(inplace=False)
# nn.Dropout(p=0.5, inplace=False)
# nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)


class SimpleNetFinal(nn.Module):
    def __init__(self):
        """
        Init function to define the layers and loss function

        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means
        nn.Conv2d(in_channels, out_channels, kernel_size=5)
        """
        super(SimpleNetFinal, self).__init__()
        self.conv_layers = nn.Sequential(
            ## TODO (2.3): Uncomment for 2.3 test
            # nn.Conv2d(1, 10, 5),
            # nn.MaxPool2d(3, 3),
            # nn.ReLU(),
            # nn.Dropout(),
            # nn.Conv2d(10, 20, 5),
            # nn.MaxPool2d(3, 3),
            # nn.ReLU()

            ## TODO: Uncomment for 2.4
            nn.Conv2d(1, 10, 5),     # 5x5 spatial support?
            # # For BatchNorm2d, Weight (gamma) = 1; Bias (beta) = 0 by default?
            # # num_features in BatchNorm2d is # of channels (why?)
            nn.BatchNorm2d(10),   # TODO: Uncomment for 2.5
            nn.MaxPool2d(2, 2),   # 3×3 window with a stride of 2.
            nn.ReLU(),

            nn.Conv2d(10, 20, 3),
            nn.BatchNorm2d(20),   # TODO: Uncomment for 2.5
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            
            nn.Dropout(), 

            nn.Conv2d(20, 20, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            # Draws weights using a uniform distribution
            # nn.Linear(in_features=500, out_features=100), # TODO: CHANGE AFTER 2.4!!
            nn.Linear(in_features=720, out_features=100),
            # https://medium.com/@vishnuam/dropout-in-convolutional-neural-networks-cnn-422a4a17da41
            nn.Linear(in_features=100, out_features=15)
        )
        # Use negative log-likelihood (good for classification w/ small number of classes)
        # (Double-check) nn.NLLLoss assigns class whereas nn.NLLLoss assigns instance
        self.loss_criterion = nn.CrossEntropyLoss(reduction="mean")

        ############################################################################
        # Student code begin
        ############################################################################
        # https://www.geeksforgeeks.org/machine-learning/dropout-in-neural-networks/

        # raise NotImplementedError(
        #     "`__init__` function in "
        #     + "`simple_net_final.py` needs to be implemented"
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
        # print("I hate myself")
        conv_output = self.conv_layers(x)
        # print(conv_output.shape)
        model_output = self.fc_layers(torch.flatten(conv_output, start_dim=1))
        return model_output
        raise NotImplementedError(
            "`forward` function in "
            + "`simple_net_final.py` needs to be implemented"
        )
        
        ############################################################################
        # Student code end
        ############################################################################

        return model_output
