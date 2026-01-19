"""Fundamental matrix utilities."""

import numpy as np


def normalize_points(points: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Perform coordinate normalization through linear transformations.
    Args:
        points: A numpy array of shape (N, 2) representing the 2D points in
            the image

    Returns:
        points_normalized: A numpy array of shape (N, 2) representing the
            normalized 2D points in the image
        T: transformation matrix representing the product of the scale and
            offset matrices
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    # Transform matrix (T) is product of scale/offset matrices
    # Calculate scale/offset matrices for T
    u, v = points[:,0], points[:,1]
    cu, cv, su, sv = np.average(u), np.average(v), 1/np.std(u),  1/np.std(v)
    scale = np.diag(np.array([su,sv,1]))
    offset = np.array([[1, 0, -cu],
                       [0, 1, -cv],
                       [0, 0,   1]])
    T = scale@offset
    # Make points homogenuous
    N = points.shape[0]
    homog_points = np.concatenate((points, np.ones((N,1))), axis=1)
    # Calculate normalized points
    normalized_homog = (T@homog_points.T).T
    # Divide by last coordinate for each point
    points_normalized = np.divide(normalized_homog, np.tile(np.reshape(normalized_homog[:,-1],(N,1)), (1,3)))
    return points_normalized[:,:2], T


    raise NotImplementedError(
        "`normalize_points` function in "
        + "`fundamental_matrix.py` needs to be implemented"
    )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return points_normalized, T


def unnormalize_F(F_norm: np.ndarray, T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """
    Adjusts F to account for normalized coordinates by using the transformation
    matrices.

    Args:
        F_norm: A numpy array of shape (3, 3) representing the normalized
            fundamental matrix
        T_a: Transformation matrix for image A
        T_B: Transformation matrix for image B

    Returns:
        F_orig: A numpy array of shape (3, 3) representing the original
            fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    # F_orig = T_b.T @ F_norm @ T_a
    return T_b.T@(F_norm@T_a)
    raise NotImplementedError(
        "`unnormalize_F` function in "
        + "`fundamental_matrix.py` needs to be implemented"
    )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F_orig


def make_singular(F_norm: np.array) -> np.ndarray:
    """
    Force F to be singular by zeroing the smallest of its singular values.
    This is done because F is not supposed to be full rank, but an inaccurate
    solution may end up as rank 3.

    Args:
    - F_norm: A numpy array of shape (3,3) representing the normalized fundamental matrix.

    Returns:
    - F_norm_s: A numpy array of shape (3, 3) representing the normalized fundamental matrix
                with only rank 2.
    """
    # Singular matrix is square matrix that is not invertible
    U, D, Vt = np.linalg.svd(F_norm)
    D[-1] = 0
    F_norm_s = np.dot(np.dot(U, np.diag(D)), Vt)

    return F_norm_s


def estimate_fundamental_matrix(
    points_a: np.ndarray, points_b: np.ndarray
) -> np.ndarray:
    """
    Calculates the fundamental matrix. You may use the normalize_points() and
    unnormalize_F() functions here. Equation (9) in the documentation indicates
    one equation of a linear system in which you'll want to solve for f_{i, j}.

    Since the matrix is defined up to a scale, many solutions exist. To constrain
    your solution, use can either use SVD and use the last Vt vector as your
    solution, or you can fix f_{3, 3} to be 1 and solve with least squares.

    Be sure to reduce the rank of your estimate - it should be rank 2. The
    make_singular() function can do this for you.

    Args:
        points_a: A numpy array of shape (N, 2) representing the 2D points in
            image A
        points_b: A numpy array of shape (N, 2) representing the 2D points in
            image B

    Returns:
        F: A numpy array of shape (3, 3) representing the fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    # Normalize points
    N = points_a.shape[0]
    normalized_a, T_a = normalize_points(points_a)
    normalized_b, T_b = normalize_points(points_b)
    # Assemble matrix A using for loop
    A = []
    for i in range(N):
        u, v = normalized_a[i]
        u_, v_ = normalized_b[i]
        A.append([u*u_, v*u_, u_, u*v_, v*v_, v_, u, v, 1])
    A = np.array(A)
    # Solve for F using SVD
    [U, S, Vt] = np.linalg.svd(A)
    F_norm = np.reshape(Vt[-1], (3,3))
    F_norm /= F_norm[2,2]
    # Solve for F using LSTSQ
    # F, _, _, _, = np.linalg.lstsq(A, np.ones(N))
    # F_norm = np.reshape(np.append(F,1), (3,3))
    # Unnormalize F and make F singular
    F = unnormalize_F(F_norm, T_a, T_b)
    return make_singular(F)
    raise NotImplementedError(
        "`estimate_fundamental_matrix` function in "
        + "`fundamental_matrix.py` needs to be implemented"
    )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F
