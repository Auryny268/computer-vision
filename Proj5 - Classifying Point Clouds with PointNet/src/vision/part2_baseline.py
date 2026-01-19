from typing import Tuple

import torch
from torch import nn


class Baseline(nn.Module):
    '''
    A simple baseline that counts points per voxel in the point cloud
    and then uses a linear classifier to make a prediction
    '''
    def __init__(self,
        classes: int,
        voxel_resolution=4,
        mode="count"
    ) -> None:
        '''
        Constructor for Baseline to define layers.

        Args:
        -   classes: Number of output classes
        -   voxel_resolution: Number of positions per dimension to count
        -   mode: Whether to count the number of points per voxel ("count") or just check binary occupancy ("occupancy")
        '''
        assert mode in ["count", "occupancy"]

        super().__init__()

        self.classifier = None
        self.voxel_resolution = None
        self.mode = None

        ############################################################################
        # Student code begin
        ############################################################################
        self.voxel_resolution = voxel_resolution
        self.mode = mode
        self.classifier = nn.Linear(voxel_resolution**3, classes)

        # raise NotImplementedError(
        #     "`__init__` function in "
        #     + "`part2_baseline.py` needs to be implemented"
        # )

        ############################################################################
        # Student code end
        ############################################################################


    def count_points(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Create the feature as input to the linear classifier by counting the number of points per voxel.
        This is effectively taking a 3D histogram for every item in a batch.

        Hint: 
        1) torch.histogramdd will be useful here

        Args:
        -   x: tensor of shape (B, N, in_dim)

        Output:
        -   counts: tensor of shape (B, voxel_resolution**3), indicating the percentage of points that landed in each voxel
        '''

        counts = None

        ############################################################################
        # Student code begin
        ############################################################################
        B,_,_ = x.shape
        bins = [self.voxel_resolution]*3
        ranges = [-self.voxel_resolution/2, self.voxel_resolution/2]*3
        # Need to pass in bins + ranges
        # Terrible idea: make for loop and then optimize
        counts = torch.zeros([B, self.voxel_resolution**3])
        for i in range(B):
            batch_counts,_ = torch.histogramdd(x[i], bins, ranges)
            batch_counts = batch_counts / batch_counts.sum()
            counts[i] = batch_counts.ravel()

        # raise NotImplementedError(
        #     "`count_points` function in "
        #     + "`part2_baseline.py` needs to be implemented"
        # )

        ############################################################################
        # Student code end
        ############################################################################

        return counts


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass of the Baseline model. Make sure you handle the case where the mode 
        is set to "occupancy" by thresholding the result of count_points on zero.

        Args:
            x: tensor of shape (B, N, 3), where B is the batch size and N is the number of points per
               point cloud
        Output:
        -   class_outputs: tensor of shape (B, classes) containing raw scores for each class
        -   None, just used to allow reuse of training infrastructure
        '''

        class_outputs = None

        ############################################################################
        # Student code begin
        ############################################################################
        counts = self.count_points(x)
        if self.mode == 'occupancy':
            counts = torch.where(counts > 0, 1.0, 0.0)
        
        class_outputs = self.classifier(counts)
        # Might have to add softmax here,,, tbd

        # raise NotImplementedError(
        #     "`forward` function in "
        #     + "`part2_baseline.py` needs to be implemented"
        # )

        
        ############################################################################
        # Student code end
        ############################################################################

        return class_outputs, None
