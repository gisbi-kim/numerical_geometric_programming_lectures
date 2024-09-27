import numpy as np


def log_map(dR):
    """
    Logarithmic map from SO(3) to the tangent space so(3).
    This function converts a rotation matrix dR to its corresponding
    skew-symmetric matrix in tangent space (so(3)).

    Parameters:
    - dR: A 3x3 rotation matrix.

    Returns:
    - A 3x3 skew-symmetric matrix representing the rotation in tangent space (so(3)).

    Explanation:
    - The log map takes a rotation matrix and computes the axis-angle representation.
    - The angle of rotation is extracted using the trace of the matrix.
    - k is a scaling factor derived from the angle of rotation, and the result is
      the scaled difference between dR and its transpose, which gives the skew-symmetric matrix.
    """
    angleaxis = np.arccos((np.trace(dR) - 1) / 2)
    if np.sin(angleaxis) == 0:
        return np.zeros((3, 3))

    k = angleaxis / (2 * np.sin(angleaxis))
    return k * (dR - dR.T)


# tangent space to rotation matrix
def exp_map(omega):
    """
    Exponential map from tangent space (so(3)) to SO(3).
    This function converts a 3D vector in tangent space (so(3)) to
    a corresponding rotation matrix in SO(3).

    Parameters:
    - omega: A 3D vector representing the skew-symmetric matrix in so(3).

    Returns:
    - A 3x3 rotation matrix in SO(3).

    Explanation:
    - The exp map takes a 3D vector (representing angular velocity) and converts it to a rotation matrix.
    - The matrix is computed using the Rodrigues' rotation formula, which is based on the rotation angle.
    - If the rotation is small (theta close to zero), the identity matrix is returned.
    """
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        return np.eye(3)
    omega_hat = omega / theta
    omega_cross = skew(omega_hat)

    return (
        np.eye(3)
        + np.sin(theta) * omega_cross
        + (1 - np.cos(theta)) * np.dot(omega_cross, omega_cross)
    )


# Convert skew-symmetric matrix to vector form
def unskew(skew_mat):
    """
    Converts a 3x3 skew-symmetric matrix to its corresponding 3D vector form.
    A skew-symmetric matrix represents a cross-product operation, and this function
    extracts the 3D vector from the matrix.

    Parameters:
    - skew_mat: A 3x3 skew-symmetric matrix.

    Returns:
    - A 3D vector corresponding to the skew-symmetric matrix.

    Explanation:
    - The function extracts the non-zero elements from the skew-symmetric matrix.
    - The elements correspond to the cross-product terms [x, y, z], which are returned as a vector.
    """
    assert skew_mat.shape == (3, 3), "Input must be a 3x3 matrix"
    x = skew_mat[2, 1]
    y = skew_mat[0, 2]
    z = skew_mat[1, 0]
    return np.array([x, y, z])


def skew(omega):
    """
    Converts a 3D vector to a 3x3 skew-symmetric matrix.
    This matrix is used to represent cross-product operations in matrix form.

    Parameters:
    - omega: A 3D vector.

    Returns:
    - A 3x3 skew-symmetric matrix.

    Explanation:
    - Given a vector omega = [x, y, z], the corresponding skew-symmetric matrix is:
      [[ 0, -z,  y],
       [ z,  0, -x],
       [-y,  x,  0]]
    """
    return np.array(
        [
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0],
        ]
    )


def oplus(T, tangent_vec):
    """
    Update SE(3) matrix T using 6D vector dxi.
    dxi[:3] represents the rotational part (in SO(3)).
    dxi[3:] represents the translational part (in R^3).

    Parameters:
    - T: (4x4) SE(3) matrix to be updated.
    - dxi: (6,) 6D vector where the first 3 elements are for rotation (SO(3))
           and the last 3 elements are for translation (R^3).

    Returns:
    - T_updated: Updated SE(3) matrix.
    """
    # Extract the rotational and translational updates
    dtheta_rot = tangent_vec[:3]  # rotational part
    dtheta_trans = tangent_vec[3:]  # translational part

    # Update the rotation (SO(3) part) using exponential map
    T[:3, :3] = T[:3, :3] @ exp_map(dtheta_rot)

    # Update the translation (R^3 part)
    T[:3, 3] += dtheta_trans

    return T
