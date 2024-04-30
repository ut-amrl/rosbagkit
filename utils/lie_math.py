import numpy as np
from scipy.spatial.transform import Rotation as R
from manifpy import SE3


def xyz_quat_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert the pose [x, y, z, qw, qx, qy, qz] to a 4x4 matrix"""
    H = np.eye(4)
    H[:3, 3] = pose[:3]
    H[:3, :3] = R.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    return H


def matrix_to_xyz_quat(matrix: np.ndarray) -> np.ndarray:
    """Convert the 4x4 matrix to [x, y, z, qw, qx, qy, qz]"""
    x, y, z = matrix[:3, 3]
    qx, qy, qz, qw = R.from_matrix(matrix[:3, :3]).as_quat()
    return np.array([x, y, z, qw, qx, qy, qz])


def xyz_quat_to_SE3(pose: np.ndarray) -> SE3:
    """Convert the pose [x, y, z, qw, qx, qy, qz] to an SE(3)"""
    position = np.array(pose[:3])
    quaternion = np.array(pose[[4, 5, 6, 3]])
    quaternion /= np.linalg.norm(quaternion)
    return SE3(position, quaternion)


def SE3_to_xyz_quat(pose: SE3) -> np.ndarray:
    """Convert the SE(3) to [x, y, z, qw, qx, qy, qz]"""
    x, y, z, qx, qy, qz, qw = pose.coeffs()
    return np.array([x, y, z, qw, qx, qy, qz])


def matrix_to_SE3(matrix: np.ndarray) -> SE3:
    """Convert the 4x4 matrix to an SE(3)"""
    trans = matrix[:3, 3]  # x, y, z
    quat = R.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]  # qw, qx, qy, qz
    return SE3(trans, quat)


def interpolate_SE3(lower_pose, upper_pose, timestamp):
    """Interpolate poses in se(3) manifold

    Args:
        lower_pose: [t1, x, y, z, qw, qx, qy, qz]
        upper_pose: [t2, x, y, z, qw, qx, qy, qz]
        timestamp (float): timestamp to interpolate

    Returns:
        interpolated_pose: [timestamp, x, y, z, qw, qx, qy, qz]
    """
    if lower_pose[0] > upper_pose[0]:
        lower_pose, upper_pose = upper_pose, lower_pose

    assert lower_pose[0] <= timestamp <= upper_pose[0]

    if lower_pose[0] == timestamp == upper_pose[0]:
        return lower_pose[1:]

    lower_SE3 = xyz_quat_to_SE3(lower_pose[1:])
    upper_SE3 = xyz_quat_to_SE3(upper_pose[1:])
    t = (timestamp - lower_pose[0]) / (upper_pose[0] - lower_pose[0])
    pose_SE3 = lower_SE3 + t * (upper_SE3 - lower_SE3)

    return SE3_to_xyz_quat(pose_SE3)
