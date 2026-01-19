import torch
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Callable

class NerfModel(nn.Module):
    
    def __init__(self, in_channels: int, filter_size: int=256):
        """This network will have a total of 8 fully connected layers. The activation function will be ReLU

        The number of input features to layer 5 will be a bit different. Refer to the docstring for the forward pass.
        Do not include an activation after layer 8 in the Sequential block. Layer 8's should output 4 features.

        Args
        ---
        in_channels (int): the number of input features from 
            the data
        filter_size (int): the number of in/out features for all layers. Layers 1 (because of in_channels), 5, and 8 are
            a bit different.
        """
        super().__init__()
        # Does activation come after each fully connected?
        self.fc_layers_group1: nn.Sequential = None  # For layers 1-3
        self.layer_4: nn.Linear = None
        self.fc_layers_group2: nn.Sequential = None  # For layers 5-8
        self.loss_criterion = None

        ##########################################################################
        # Student code begins here
        ##########################################################################
        self.fc_layers_group1 = nn.Sequential( # For layers 1-3
            nn.Linear(in_channels, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, filter_size),
            nn.ReLU()
        ) 
        self.layer_4 = nn.Linear(filter_size, filter_size)
        self.fc_layers_group2 = nn.Sequential( # For layers 5-8
            nn.Linear(2*filter_size, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, 4)
        )

        self.loss_criterion = nn.MSELoss()
        # raise NotImplementedError('`init` function in `NerfModel` needs to be implemented')

        ##########################################################################
        # Student code ends here
        ##########################################################################
  
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass of the model. 
        
        NOTE: The input to layer 5 should be the concatenation of post-activation values from layer 4 with 
        post-activation values from layer 3. Therefore, be extra careful about how self.layer_4 is used, the order of
        concatenation, and what the specified input shape to layer 5 should be. The output from layer 5 and the 
        dimensions thereafter should be filter_size.
        
        Args
        ---
        x (torch.Tensor): input of shape 
            (batch_size, in_channels)
        
        Returns
        ---
        rgb (torch.Tensor): The predicted rgb values with 
            shape (batch_size, 3)
        sigma (torch.Tensor): The predicted density values with shape (batch_size)
        """
        rgb = None
        sigma = None

        ##########################################################################
        # Student code begins here
        ##########################################################################
        # print(x.shape)
        # print("test4.1")
        layer3_output = self.fc_layers_group1(x)
        # print("test4.2")
        layer4_output = F.relu(self.layer_4(layer3_output))
        # print("test4.3")
        # print(layer3_output.shape)
        # print(layer4_output.shape)
        output = torch.cat((layer4_output, layer3_output), dim=-1)
        # print(output.shape)
        output = self.fc_layers_group2(output)
        # print(output.shape)
        # raise NotImplementedError('`forward` function in `NerfModel` needs to be implemented')
        # print("test4.4")
        rgb,sigma = torch.sigmoid(output[:,:,:3]), F.relu(output[:,:,3])
        ##########################################################################
        # Student code ends here
        ##########################################################################

        return rgb, sigma

def get_rays(height: int, width: int, intrinsics: torch.Tensor, tform_cam2world: torch.Tensor) \
    -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the origin and direction of rays passing through all pixels of an image (one ray per pixel).
    
    Args
    ---
    height (int): 
        the height of an image.
    width (int): the width of an image.
    intrinsics (torch.Tensor): Camera intrinsics matrix of shape (3, 3).
    tform_cam2world (torch.Tensor): A 6-DoF rigid-body transform (shape: :math:`(4, 4)`) that
        transforms a 3D point from the camera coordinate space to the world frame coordinate space.
    
    Returns
    ---
    ray_origins (torch.Tensor): A tensor of shape :math:`(height, width, 3)` denoting the centers of
        each ray. Note that desipte that all ray share the same origin, 
        here we ask you to return the ray origin for each ray as (height, width, 3).
    ray_directions (torch.Tensor): A tensor of shape :math:`(height, width, 3)` denoting the
        direction of each ray.
    """
    device = tform_cam2world.device
    ray_directions = torch.zeros((height, width, 3), device=device)  # placeholder
    ray_origins = torch.zeros((height, width, 3), device=device)  # placeholder

    ##########################################################################
    # Student code begins here
    ##########################################################################
    # Camera Intrinsics matrix has camera center baked in
    

    # Steps:
    # 0. Generate pixel coordinate first -> then intrinsic will take care bc inverse K will shift everything to correct "pov"
    # print(f"height: {height};\twidth: {width}")
    # print(f"intrinsics:\n{intrinsics}")
    # print(f"tform_cam2world:\n{tform_cam2world}")

    x_grid = torch.arange(width, device=device).repeat(height, 1)
    y_grid = torch.arange(height, device=device).repeat(width, 1).t()

    # Create a tensor of ones with the same shape
    ones = torch.ones((height, width), device=device)

    # Stack the x, y, and ones tensors along a new dimension to get (H, W, 3)
    # Note: the order is (x, y, 1) to align with standard computer graphics conventions
    # pixel_coord is transposed to make up for the order later in matmul
    pixel_coord = torch.stack([x_grid, y_grid, ones], dim=-1)
    # print(f"pixel_coord: {pixel_coord.shape}\n{pixel_coord}")

    # 1. Multiply homog img points (3d) by inverted intrinsics matrix
    # BE CAREFUL ABT ORDER!
    k_inv = torch.inverse(intrinsics)
    camera_coord = pixel_coord @ k_inv.T
    # print(f"camera_coord: {camera_coord.shape}\n{camera_coord}")

    # 2. Extract Rotation Matrix and Translation Vector
    rot_matrix = tform_cam2world[:3, :3]
    t_vector = tform_cam2world[:3, 3]
    # print(f"rot_matrix:\n{rot_matrix}")
    # print(f"t_vector:\n{t_vector}")
    
    ray_directions = camera_coord @ rot_matrix.T
    ray_origins = t_vector.expand(height, width, 3)
   
    # print(f"predicted ray origins:\n{ray_origins}")
    # print(f"predicted ray directions:\n{ray_directions}")
    return ray_origins, ray_directions



    # raise NotImplementedError('`get_rays()` function needs to be implemented')

    ##########################################################################
    # Student code ends here
    ##########################################################################

    return ray_origins, ray_directions

def sample_points_from_rays(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    near_thresh: float,
    far_thresh: float,
    num_samples: int,
    randomize:bool = True
) -> tuple[torch.tensor, torch.tensor]:
    """Sample 3D points on the given rays. The near_thresh and far_thresh
    variables indicate the bounds of sampling range.
    
    Args
    ---
    ray_origins (torch.Tensor): Origin of each ray in the "bundle" as returned by the
        `get_rays` method (shape: :math:`(height, width, 3)`).
    ray_directions (torch.Tensor): Direction of each ray in the "bundle" as returned by the
        `get_rays` method (shape: :math:`(height, width, 3)`).
    near_thresh (float): The 'near' extent of the bounding volume (i.e., the nearest depth
        coordinate that is of interest/relevance).
    far_thresh (float): The 'far' extent of the bounding volume (i.e., the farthest depth
        coordinate that is of interest/relevance).
    num_samples (int): Number of samples to be drawn along each ray. Samples are drawn
        randomly, whilst trying to ensure "some form of" uniform spacing among them.
    randomize (optional, bool): Whether or not to randomize the sampling of query points.
        By default, this is set to `True`. If disabled (by setting to `False`), we sample
        uniformly spaced points along each ray (i.e., the lower bound of each bin).
    
    Returns
    ---
    query_points (torch.Tensor): Query 3D points along each ray
        (shape: :math:`(height, width, num_samples, 3)`).
    depth_values (torch.Tensor): Sampled depth values along each ray
        (shape: :math:`(height, width, num_samples)`).
    """
    # print(f"ray o/d: {ray_directions.shape}")
    # print(f"num_samples: {num_samples}")

    device = ray_origins.device
    height, width = ray_origins.shape[:2]
    depth_values = torch.zeros((height, width, num_samples), device=device) # placeholder
    query_points = torch.zeros((height, width, num_samples, 3), device=device) # placeholder
    
    ##########################################################################
    # Student code begins here
    ##########################################################################
    # For each ray we want to sample num_sample amt of pts
    # To travel along each ray, we have t_i * direction
    # print(f"near:{near_thresh}, far:{far_thresh}")
    thresh_range = far_thresh - near_thresh
    # Using the equation, you want to do a for loop (maybe?) if randomly sampling
    # NVM! Can just randomize if needed -> start by getting lower bounds
    # Should determine if randomization happens for each ray individually or as a whole for all rays
    bins = torch.arange(num_samples, device=device)
    if randomize:
        # randomly generate num_samples sized vector, make sure it's in range and then add to depth_values
        bins = bins + torch.rand(num_samples)
    depth_values = thresh_range/num_samples * bins + near_thresh
    base_origins = ray_origins.unsqueeze(2).repeat((1,1,num_samples,1))
    base_directions = ray_directions.unsqueeze(2).repeat((1,1,num_samples,1))
    # print(f"base points:{base_origins.shape}")
    # print(depth_values.repeat(3,1).T * base_directions)
    query_points = base_origins + depth_values.repeat(3,1).T * base_directions
    # print(query_points.shape)
    # raise NotImplementedError('`sample_points_from_rays()` function needs to be implemented')

    ##########################################################################
    # Student code ends here
    ##########################################################################
    
    # print(f"actual query_points:\n{query_points}")
    # print(f"actual depth_values:\n{depth_values}")
    return query_points, depth_values

def cumprod_exclusive(x: torch.tensor) -> torch.tensor:
    """ Helper function that computes the cumulative product of the input tensor, excluding the current element
    Example:
    > cumprod_exclusive(torch.tensor([1,2,3,4,5]))
    > tensor([ 1,  1,  2,  6, 24])
    
    Args:
    -   x: Tensor of length N
    
    Returns:
    -   cumprod: Tensor of length N containing the cumulative product of the tensor
    """

    cumprod = torch.cumprod(x, -1)
    cumprod = torch.roll(cumprod, 1, -1)
    cumprod[..., 0] = 1.
    return cumprod

def compute_compositing_weights(sigma: torch.Tensor, depth_values: torch.Tensor) -> torch.Tensor:
    """This function will compute the compositing weight for each query point.

    Args
    ---
    sigma (torch.Tensor): Volume density at each query location (X, Y, Z)
        (shape: :math:`(height, width, num_samples)`).
    depth_values (torch.Tensor): Sampled depth values along each ray
        (shape: :math:`(height, width, num_samples)`).
    
    Returns:
    weights (torch.Tensor): Rendered compositing weight of each sampled point 
        (shape: :math:`(height, width, num_samples)`).
    """

    device = depth_values.device
    weights = torch.ones_like(sigma, device=device) # placeholder

    ##########################################################################
    # Student code begins here
    ##########################################################################

     # Jesus christ we have to do some quadrature
    # DOUBLE QUADRATURE yay.... (not!)
    # 1. Start by getting the Ti -> equation looks a little fucked tbh

    # Probably want to start by generating deltas
    # delta = tensor of distances btwn entries  # Should see if there's a np or pytorch function for this
    # print(depth_values)
    delta = depth_values[:,:,1:] - depth_values[:,:,:-1]
    # Need to replace last entry with 1e9
    # *depth_values.shape[:2] alegedly unpacks list
    ones = torch.ones((*depth_values.shape[:2], 1), device=device) * 1e9
    delta = torch.cat((delta, ones), dim=-1)
    # print(deltas)

    # After that, do elementwise multiplication of delta and sigma (deltas should scale sigmas)
    # Will need to save this separate for compositing weights
    product = sigma * delta

    # Could do cumprod with negative expotentials instead of cumsum (depending on how cumsum works)
    T_ = cumprod_exclusive(torch.exp(-product))

    # Afterwards calculating the weights should be easy
    weights = T_ * (-torch.exp(-product) + 1)

    # raise NotImplementedError('`compute_compositing_weights()` function needs to be implemented')

    ##########################################################################
    # Student code ends here
    ##########################################################################

    return weights

def get_minibatches(inputs: torch.Tensor, chunksize: int = 1024 * 32) -> list[torch.Tensor]:
    """Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
    Each element of the list (except possibly the last) has dimension `0` of length
    `chunksize`.
    """
    return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

def render_image_nerf(height: int, width: int, intrinsics: torch.tensor, tform_cam2world: torch.tensor,
                      near_thresh: float, far_thresh: float, depth_samples_per_ray: int,
                      encoding_function: Callable, model:NerfModel, rand:bool=False) \
                      -> tuple[torch.Tensor, torch.Tensor]:
    """ This function will utilize all the other rendering functions that have been implemented in order to sample rays,
    pass those rays to the NeRF model to get color and density predictions, and then use volume rendering to create
    an image of this view. 

    Hints: 
    ---
    It is a good idea to "flatten" the height/width dimensions of the data when passing to the NeRF (maintain the color
    channel dimension) and then "unflatten" the outputs. 
    To avoid running into memory limits, it's recommended to use the given get_minibatches() helper function to 
    divide up the input into chunks. For each minibatch, supply them to the model and then concatenate the corresponding
    output vectors from each minibatch to form the complete outpute vectors. 
    
    Args
    ---
    height (int): 
        the pixel height of an image.
    width (int): the pixel width of an image.
    intrinsics (torch.tensor): Camera intrinsics matrix of shape (3, 3).
    tform_cam2world (torch.Tensor): A 6-DoF rigid-body transform (shape: :math:`(4, 4)`) that
        transforms a 3D point from the camera coordinate space to the world frame coordinate space.
    near_thresh (float): The 'near' extent of the bounding volume (i.e., the nearest depth
        coordinate that is of interest/relevance).
    far_thresh (float): The 'far' extent of the bounding volume (i.e., the farthest depth
        coordinate that is of interest/relevance).
    depth_samples_per_ray (int): Number of samples to be drawn along each ray. Samples are drawn
        randomly, whilst trying to ensure "some form of" uniform spacing among them.
    encoding_function (Callable): The function used to encode the query points (e.g. positional encoding)
    model (NerfModel): The NeRF model that will be used to render this image
    randomize (optional, bool): Whether or not to randomize the sampling of query points.
        By default, this is set to `True`. If disabled (by setting to `False`), we sample
        uniformly spaced points along each ray (i.e., the lower bound of each bin).
    
    Returns
    ---
    rgb_predicted (torch.tensor): 
        A tensor of shape (height, width, num_channels) with the color info at each pixel.
    depth_predicted (torch.tensor): A tensor of shape (height, width) containing the depth from the camera at each pixel.
    """

    rgb_predicted, depth_predicted = None, None

    ##########################################################################
    # Student code begins here
    ##########################################################################
    # 1. Compute camera rays
    # print("test1")
    ray_o, ray_d = get_rays(height, width, intrinsics, tform_cam2world)
    # 2. Sample 3D points on rays
    # print("test2")
    query_points, depth_values = sample_points_from_rays(ray_o, ray_d, 
                                                         near_thresh, far_thresh,
                                                         depth_samples_per_ray, rand
                                                         ) 
    # 3. Positionally encode sampled points
    # Q: I'm guessing we only have to encode query points but idk for sure
    # print("test3")
    # print(f"query_points:{query_points.shape}")
    encoded_pts = encoding_function(query_points)
    # print(f"encoded_points:{encoded_pts.shape}")
    # 4. Feed encoded points into model to compute color and volume density
    """
     Hints: 
    ---
    It is a good idea to "flatten" the height/width dimensions of the data when passing to the NeRF (maintain the color
    channel dimension) and then "unflatten" the outputs. 
    To avoid running into memory limits, it's recommended to use the given get_minibatches() helper function to 
    divide up the input into chunks. For each minibatch, supply them to the model and then concatenate the corresponding
    output vectors from each minibatch to form the complete outpute vectors.
    """ 
    encoded_pts = torch.flatten(encoded_pts, end_dim=1)
    minibatches = get_minibatches(encoded_pts)
    # print("test4")
    rgb, sigma = [],[]
    for batch in minibatches:
        batch_rgb, batch_sigma= model.forward(batch)
        # print(f"batch rgb:{batch_rgb.shape}; batch sigma:{batch_sigma.shape}")
        rgb.append(batch_rgb)
        sigma.append(batch_sigma)
    rgb = torch.reshape(torch.cat(rgb), (height, width, depth_samples_per_ray, 3))
    sigma = torch.reshape(torch.cat(sigma), (height, width, depth_samples_per_ray))
    # print(f"rgb:{rgb.shape}; sigma:{sigma.shape}")

    # 5. Use density to compute compositing weights of samples on a ray
    # print("test5")
    
    # print(f"depth_values:{depth_values.shape}")
    depth_values = depth_values.repeat(height, width, 1)
    # print(f"depth_values:{depth_values.shape}")
    weights = compute_compositing_weights(sigma, depth_values)

    # 6. Compute color and depth maps using weighted sum using compositing weights
    # print(f"weights:{weights.shape}")
    # print(f"rgb:{rgb.shape}; sigma:{sigma.shape}")
    depth_predicted = torch.sum(weights * sigma, dim=2)
    # TODO: Figure out the formula for this once I land?
    weights = weights.unsqueeze(3).repeat((1,1,1,3))
    # print(f"weights:{weights.shape}")
    rgb_predicted = torch.sum(weights * rgb, dim=2)
    

    # raise NotImplementedError('`render_image_nerf()` function needs to be implemented')
    
    ##########################################################################
    # Student code ends here
    ##########################################################################

    return rgb_predicted, depth_predicted