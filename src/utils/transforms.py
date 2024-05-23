import numpy as np
from scipy.spatial.transform import Rotation as R


def xyz_quat_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert the pose [x, y, z, qw, qx, qy, qz] to a 4x4 matrix"""
    matrix = np.eye(4)
    matrix[:3, 3] = pose[:3]
    matrix[:3, :3] = R.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    return matrix


def matrix_to_xyz_quat(matrix: np.ndarray) -> np.ndarray:
    """Convert the 4x4 matrix to [x, y, z, qw, qx, qy, qz]"""
    x, y, z = matrix[:3, 3]
    qx, qy, qz, qw = R.from_matrix(matrix[:3, :3]).as_quat()
    return np.array([x, y, z, qw, qx, qy, qz])
