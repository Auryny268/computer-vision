import math

import numpy as np
import cv2


def calculate_num_ransac_iterations(
    prob_success: float, sample_size: int, ind_prob_correct: int
) -> int:
    """
    Calculate the number of RANSAC iterations needed for a given guarantee of success.

    Args:
    -   prob_success: float representing the desired guarantee of success
    -   sample_size: int the number of samples included in each RANSAC iteration
    -   ind_prob_success: float representing the probability that each element in a sample is correct

    Returns:
    -   num_samples: int the number of RANSAC iterations needed

    """
    num_samples = None
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    # Outlier ratio same as `ind_prob_success`??
    num_samples = np.log(1-prob_success) / np.log(1 - (ind_prob_correct)**sample_size)
    return int(num_samples)
    raise NotImplementedError(
        "`calculate_num_ransac_iterations` function in "
        + "`ransac.py` needs to be implemented"
    )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return int(num_samples)

def ransac_homography(
    points_a: np.ndarray, points_b: np.ndarray
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Uses the RANSAC algorithm to robustly estimate a homography matrix.

    Args:
    -   points_a: A numpy array of shape (N, 2) of points from image A.
    -   points_b: A numpy array of shape (N, 2) of corresponding points from image B.

    Returns:
    -   best_H: The best homography matrix of shape (3, 3).
    -   inliers_a: The subset of points_a that are inliers (M, 2).
    -   inliers_b: The subset of points_b that are inliers (M, 2).
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    #                                                                         #
    # HINT: You are allowed to use the `cv2.findHomography` function to       #
    # compute the homography from a sample of points. To compute a direct     #
    # solution without OpenCV's built-in RANSAC, use it like this:            #
    #   H, _ = cv2.findHomography(sample_a, sample_b, 0)                      #
    # The `0` flag ensures it computes a direct least-squares solution.       #
    ###########################################################################

    # https://www.geeksforgeeks.org/computer-vision/what-is-homography-how-to-estimate-homography-between-two-images/
    
    N = points_a.shape[0]
    # Calculated my number of iterations
    num_iterations = calculate_num_ransac_iterations(0.999, 4, 0.18)  # Guess percentage of correct points
    print(f"num_iterations = {num_iterations}")
    x, x_prime = np.concatenate((points_a, np.ones((N,1))), axis=1), np.concatenate((points_b, np.ones((N,1))), axis=1)
    best_H, inliers_a, inliers_b = None, np.array([]), np.array([])
    threshold = 5 # idk play with it or smth
    for i in range(num_iterations):
        # Choose 4 points at random
        ransac_ind = np.random.randint(0, N, size=4)
        sample_a, sample_b = points_a[ransac_ind], points_b[ransac_ind]
        # Find homography H _. shape (3,3) & find Hx (w/ some adjustments)
        H, _ = cv2.findHomography(sample_a, sample_b, 0)
        Hx = (H@x.T).T
        Hx = np.divide(Hx, np.tile(np.reshape(Hx[:,2],(N,1)), (1,3)))
        # Take distances from expected points (x')
        dists = np.linalg.norm((Hx-x_prime), axis=1)
        # Count inliers
        inlier_ind = np.array(np.nonzero(dists < threshold)).ravel()
        if inlier_ind.shape[0] > inliers_a.shape[0]:
            best_H, inliers_a, inliers_b = H, points_a[inlier_ind], points_b[inlier_ind]
    print(f"best_H {best_H}, # of inliers: {inliers_a.shape[0]}")
    return best_H, inliers_a, inliers_b

    raise NotImplementedError(
        "`ransac_homography` function in "
        + "`part6_ransac.py` needs to be implemented"
    )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return best_H, inliers_a, inliers_b
