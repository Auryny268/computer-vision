import glob
import os
from typing import Tuple
# import torch
# import torchvision.transforms as transforms
# from vision.image_loader import ImageLoader
# from vision.data_transforms import get_fundamental_transforms
import numpy as np
from PIL import Image


def compute_mean_and_std(dir_name: str) -> Tuple[float, float]:
    """Compute the mean and the standard deviation of all images present within the directory.

    Note: convert the image in grayscale and then in [0,1] before computing mean
    and standard deviation

    Mean = (1/n) * sum_{i=1}^n x_i
    Variance = (1 / (n-1)) * sum_{i=1}^n (x_i - mean)^2
    Standard Deviation = sqrt(Variance)

    Args:
        dir_name: the path of the root dir

    Returns:
        mean: mean value of the gray color channel for the dataset
        std: standard deviation of the gray color channel for the dataset
    """
    mean = 0.0
    variance = 0.0
    std = 0.0

    ############################################################################
    # Student code begin
    ############################################################################
    # ChatGPT Code -> os.path.join makes a path out of all inputs
    # glob.glob find all path names matching a specified pattern (where '*' is "wildcard")
    # instead of trying to do everything in for loop, keep track of total pixels
    
    # image_paths = glob.glob(os.path.join(dir_name, "*", "*", "*"))
    image_paths = glob.glob(os.path.join(dir_name, "**/*.jpg"), recursive=True)
    images = [np.ravel(np.asarray(Image.open(f).convert('L'), dtype=np.float32) / 255.0) for f in image_paths]
    # stacks arrays along column (i.e. rows get stacked)
    pixels = np.concatenate(images)
    mean = np.average(pixels)
    variance = np.var(pixels, ddof=1)
    std = np.sqrt(variance)
    return mean, std










    # New idea -> do this with torch/torchvision transforms
    # inp_size = (64,64)  # Try different combos and see what passes the test
    # print("test1")
    # transformmem = transforms.Compose([
    #     transforms.Grayscale(),
    #     transforms.Resize(inp_size),
    #     transforms.ToTensor(),
    # ])
    # print("test1")
    # image_set = ImageLoader(dir_name, split="train", transform=transformmem)
    # var, mean = torch.var_mean(image_set.dataset)
    # std = np.sqrt(var)
    # return mean, std


    # # ChatGPT solution
    # sum_, sum_sq, total = 0.0, 0.0, 0
    # for f in image_paths:
    #     # Explicit float conversion BEFORE division
    #     grayscale_img = Image.open(f).convert("L").resize((224,224))
    #     img = np.asarray(grayscale_img, dtype=np.float32) / 255.0
    #     sum_ += np.sum(img)
    #     sum_sq += np.sum(img ** 2)
    #     total += img.size

    # mean = sum_ / total
    # # ✅ Match population variance (ddof=0) for reference consistency
    # variance = (sum_sq / total) - (mean ** 2)
    # std = np.sqrt(variance)
    # print(f"mean:{mean:.8f}, std:{std:.8f}")
    # return mean, std

    raise NotImplementedError(
            "`compute_mean_and_std` function in "
            + "`stats_helper.py` needs to be implemented"
        )

    ############################################################################
    # Student code end
    ############################################################################
    return mean, std

