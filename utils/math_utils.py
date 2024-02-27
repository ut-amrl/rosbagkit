"""
Author:      Dongmyeong Lee (domlee[at]utexas.edu)
Date:        Nov 22, 2023
Description: functions for math operations
"""
from typing import List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

def average_rpy(angles: np.ndarray, degrees: bool = False) -> np.ndarray:
    """
    Average rotatmion matrix and compute SVD for the average rotation matrix
    to represent geometric mean of rotation matrices
    """
    sum_R = np.zeros((3, 3))
    for angle in angles:
        sum_R += R.from_euler("xyz", angle, degrees=degrees).as_matrix()
    U, _, Vt = np.linalg.svd(sum_R)
    return R.from_matrix(U @ Vt).as_euler("xyz", degrees=degrees)


def solve_linear_quadratic(G: np.ndarray, l: np.ndarray) -> np.ndarray:
    """
    Solve linear-quadratic equation
    ref: https://math.stackexchange.com/questions/679622/intersection-between-conic-and-line-in-homogeneous-space

    Args:
        G: (3, 3) Dual Conic Matrix (adjugate of Conic Matrix)
           G = [a, c, d]
               [c, b, e]
               [d, e, f]

        l: (3,) Linear Term (ax + by + c = 0)
    Returns:
        intersection: (N, 2) Intersection Points
    """
    # If l is tangent to the conic there are only one intersection
    if np.isclose(l @ G @ l.T, 0.0):
        q = np.dot(G, l)
        q = q[:2] / q[2]
        return q

    # If l is not intersecting with the conic
    if l @ G @ l.T < 0.0:
        return np.empty((0, 2))

    # There are two intersections
    a = G[0, 0]
    b = G[1, 1]
    c = G[0, 1]
    d = G[0, 2]
    e = G[1, 2]
    f = G[2, 2]

    p = np.dot(G, l)  # [u, v, w]   p
    u = p[0]
    v = p[1]
    w = p[2]

    k_0 = a * w**2 - 2 * d * u * w + f * u**2
    k_1 = -2 * (c * w**2 - w * (d * v + e * u) + f * u * v)
    k_2 = w**2 * (b - a) + 2 * w * (d * u - e * v) + f * (v**2 - u**2)

    k_3 = (2 * k_0 + k_2) / np.sqrt(k_1**2 + k_2**2)
    k_3 = np.clip(k_3, -1.0, 1.0)

    phi_1 = np.arctan2(k_1, k_2) / 2 - np.arcsin(k_3) / 2 - np.pi / 4
    phi_2 = np.arctan2(k_1, k_2) / 2 + np.arcsin(k_3) / 2 + np.pi / 4

    q_1 = np.array(
        [
            (c * w - d * v) * np.cos(phi_1) + (d * u - a * w) * np.sin(phi_1),
            (b * w - e * v) * np.cos(phi_1) + (e * u - c * w) * np.sin(phi_1),
            (e * w - f * v) * np.cos(phi_1) + (f * u - d * w) * np.sin(phi_1),
        ]
    )
    q_2 = np.array(
        [
            (c * w - d * v) * np.cos(phi_2) + (d * u - a * w) * np.sin(phi_2),
            (b * w - e * v) * np.cos(phi_2) + (e * u - c * w) * np.sin(phi_2),
            (e * w - f * v) * np.cos(phi_2) + (f * u - d * w) * np.sin(phi_2),
        ]
    )

    q_1 = q_1[:2] / q_1[2]
    q_2 = q_2[:2] / q_2[2]
    return np.array([q_1, q_2])


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
