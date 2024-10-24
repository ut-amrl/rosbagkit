import numpy as np
from scipy.spatial.transform import Rotation as R
from manifpy import SE3, SO3, SE3Tangent, SO3Tangent


def xyz_quat_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert the pose [x, y, z, qw, qx, qy, qz] to a 4x4 matrix"""
    assert pose.shape == (7,), f"{pose.shape} != (7,)"
    H = np.eye(4)
    H[:3, 3] = pose[:3]
    H[:3, :3] = R.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    return H


def matrix_to_xyz_quat(matrix: np.ndarray) -> np.ndarray:
    """Convert the 4x4 matrix to [x, y, z, qw, qx, qy, qz]"""
    assert matrix.shape == (4, 4), f"{matrix.shape} != (4, 4)"
    x, y, z = matrix[:3, 3]
    qx, qy, qz, qw = R.from_matrix(matrix[:3, :3]).as_quat()
    return np.array([x, y, z, qw, qx, qy, qz])


def xyz_rpy_to_matrix(pose: np.ndarray, degrees: bool = False) -> np.ndarray:
    """Convert the pose [x, y, z, roll, pitch, yaw] to a 4x4 matrix"""
    assert pose.shape == (6,), f"{pose.shape} != (6,)"
    H = np.eye(4)
    H[:3, 3] = pose[:3]
    H[:3, :3] = R.from_euler("xyz", pose[3:], degrees=degrees).as_matrix()
    return H


def xyz_quat_to_SE3(pose: np.ndarray) -> SE3:
    """Convert the pose [x, y, z, qw, qx, qy, qz] to an SE(3)"""
    assert pose.shape == (7,), f"{pose.shape} != (7,)"
    position = np.array(pose[:3])  # [x, y, z]
    quaternion = np.array(pose[[4, 5, 6, 3]], dtype=np.float64)  # [qx, qy, qz, qw]
    quaternion /= np.linalg.norm(quaternion)
    return SE3(position, quaternion)


def SE3_to_xyz_quat(pose: SE3) -> np.ndarray:
    """Convert the SE(3) to [x, y, z, qw, qx, qy, qz]"""
    x, y, z, qx, qy, qz, qw = pose.coeffs()
    return np.array([x, y, z, qw, qx, qy, qz])


def matrix_to_SE3(matrix: np.ndarray) -> SE3:
    """Convert the 4x4 matrix to an SE(3)"""
    assert matrix.shape == (4, 4), f"{matrix.shape} != (4, 4)"
    trans = matrix[:3, 3]  # [x, y, z]
    quat = R.from_matrix(matrix[:3, :3]).as_quat()  # [qx, qy, qz, qw]
    return SE3(trans, quat)
