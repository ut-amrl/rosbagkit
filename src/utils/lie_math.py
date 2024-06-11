import numpy as np
from typing import List
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


def average_SO3(rotations: List[np.ndarray]) -> np.ndarray:
    """
    Geodesic L2 single rotation averaging algorithm
    Reference: Rotation Averaging [Richard Hartley et.al., 2013]
    """
    assert len(rotations) > 0
    assert all(rot.shape == (3, 3) for rot in rotations)
    assert all(np.isclose(np.linalg.det(rot), 1.0) for rot in rotations)
    assert all(np.allclose(rot.T @ rot, np.eye(3)) for rot in rotations)

    avg_R = rotations[0]

    for trial in range(100):
        avg_residual_log = SO3Tangent(np.zeros(3))
        for rot in rotations:
            residual = R.from_matrix(avg_R.T @ rot).as_quat()
            residual_log = SO3(residual).log()
            avg_residual_log += residual_log
        avg_residual_log /= len(rotations)

        if np.allclose(avg_residual_log.coeffs(), np.zeros(3)):
            break

        avg_R = avg_R @ avg_residual_log.exp().rotation()

    return avg_R


def average_rpy(rotations: List[np.ndarray], degrees=False) -> np.ndarray:
    """
    Geodesic L2 single rotation averaging algorithm for roll, pitch, yaw
    """
    assert len(rotations) > 0
    assert all(rot.shape == (3,) for rot in rotations)

    SO3_rotations = [
        R.from_euler("xyz", rot, degrees=degrees).as_matrix() for rot in rotations
    ]

    avg_SO3 = average_SO3(SO3_rotations)

    return R.from_matrix(avg_SO3).as_euler("xyz", degrees=degrees)


if __name__ == "__main__":
    # X = SO3.Random()
    # rotations = []
    # for i in range(10):
    #     w = SO3Tangent(np.random.normal(0, 0.01, 3))
    #     rotations.append((X + w).rotation())

    # avg_R = average_SO3(rotations)
    # print(avg_R)
    # print(X.rotation())

    # rpy = average_rpy([R.from_matrix(rot).as_euler("xyz") for rot in rotations])
    # print(R.from_euler("xyz", rpy).as_matrix())

    X = SE3.Random()
    pose = SE3_to_xyz_quat(X)
    print(pose)

    pose_matrix = xyz_quat_to_matrix(pose)
    print(pose_matrix)

    print(SE3_to_xyz_quat(matrix_to_SE3(pose_matrix)))
    print(xyz_quat_to_matrix(SE3_to_xyz_quat(matrix_to_SE3(pose_matrix))))

    pose_SE3 = xyz_quat_to_SE3(pose)
    pose_SE3_inv = pose_SE3.inverse()
    pose_inv = SE3_to_xyz_quat(pose_SE3_inv)
    print(pose_inv)
    print(xyz_quat_to_matrix(pose_inv))
