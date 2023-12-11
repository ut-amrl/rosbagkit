"""
Author:      Dongmyeong Lee (domlee[at]utexas.edu)
Date:        Nov 22, 2023
Description: functions for math operations
"""
from typing import List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from manifpy import SO3, SO3Tangent


def average_rpy(angles: np.ndarray, degrees: bool = False) -> np.ndarray:
    """
    Average roll, pitch, yaw angles

    Args:
        angles: (N, 3) array of roll, pitch, yaw angles
        degrees: if True, angles are in degrees otherwise radians

    Returns:
        average_angle: (3,) array of average roll, pitch, yaw angles
    """
    xi_list = []
    for angle in angles:
        quaternion = R.from_euler("zyx", angle, degrees=degrees).as_quat()  # x, y, z, w
        xi = SO3(quaternion).log().coeffs()
        xi_list.append(xi)

    xi_avg = SO3Tangent(np.mean(xi_list, axis=0))
    avg_quat = xi_avg.exp().quat()

    return R.from_quat(avg_quat).as_euler("zyx", degrees=degrees)


def cofactor(A):
    """
    Calculate cofactor matrix of A

    Args:
        A: (N, N) array

    Returns:
        C: (N, N) array of cofactor matrix
    """
    sel_rows = np.ones(A.shape[0], dtype=bool)
    sel_columns = np.ones(A.shape[1], dtype=bool)
    C = np.zeros_like(A)
    sgn_row = 1
    for row in range(A.shape[0]):
        # Unselect current row
        sel_rows[row] = False
        sgn_col = 1
        for col in range(A.shape[1]):
            # Unselect current column
            sel_columns[col] = False
            # Extract submatrix
            MATij = A[sel_rows][:, sel_columns]
            C[row, col] = sgn_row * sgn_col * np.linalg.det(MATij)
            # Reselect current column
            sel_columns[col] = True
            sgn_col = -sgn_col
        sel_rows[row] = True
        # Reselect current row
        sgn_row = -sgn_row
    return C


def adjugate(A):
    """
    Calculate adjugate matrix of A
    """
    return cofactor(A).T
