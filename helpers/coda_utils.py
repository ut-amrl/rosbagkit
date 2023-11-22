import yaml
import numpy as np

from typing import Tuple, Dict
from scipy.spatial.transform import Rotation


def load_extrinsic_matrix(extrinsic_file: str) -> np.ndarray:
    """
    Load extrinsic matrix from a yaml file.

    Args:
        extrinsic_file: Path to the yaml file containing the extrinsic matrix.

    Returns:
        extrinsic_matrix: (4, 4) extrinsic matrix (homogeneous coordinates
    """
    with open(extrinsic_file, "r") as f:
        params = yaml.safe_load(f)["extrinsic_matrix"]

        if "R" in params.keys() and "T" in params.keys():
            extrinsic_matrix = np.eye(4)
            extrinsic_matrix[:3, :3] = np.array(params["R"]["data"]).reshape(
                params["rows"], params["cols"]
            )
            extrinsic_matrix[:3, 3] = np.array(params["T"])
        else:
            extrinsic_matrix = np.array(params["data"]).reshape(
                params["rows"], params["cols"]
            )
    return extrinsic_matrix


def load_camera_params(intrinsic_file: str) -> Dict[str, np.ndarray]:
    """
    Load camera parameters from a yaml file.

    Args:
        intrinsic_file: Path to the yaml file containing the camera parameters.

    Returns:
        K: (3, 3) intrinsic matrix
        image_size: (2,) image size
        D: (5,) distortion coefficients
    """
    with open(intrinsic_file, "r") as f:
        params = yaml.safe_load(f)
        matrix_params = params["camera_matrix"]

        intrinsic_matrix = np.array(matrix_params["data"]).reshape(
            matrix_params["rows"], matrix_params["cols"]
        )
        image_size = np.array([params["image_width"], params["image_height"]])
        distortion_coeffs = np.array(params["distortion_coefficients"]["data"])
    return {"K": intrinsic_matrix, "image_size": image_size, "D": distortion_coeffs}


def load_keyframe_pose(keyframe_pose_file: str) -> np.ndarray:
    """
    Load estimated pose from a keyframe file (.data) from interactive_slam

    Args:
        keyframe_pose_file: Path to the (.data) file containing the estimated pose.

    Returns:
        keyframe_pose: (8,) estimated pose (timestamp, x, y, z, qw, qx, qy, qz)
    """
    with open(keyframe_pose_file, "r") as f:
        lines = f.readlines()

        # timestamp
        timestamp_line = lines[0].strip().split(" ")
        timestamp = float(timestamp_line[1])

        # estimated pose (SE3)
        pose_lines = lines[2:6]
        keyframe_pose_matrix = np.array(
            [list(map(float, line.split())) for line in pose_lines]
        )
        r = Rotation.from_matrix(keyframe_pose_matrix[:3, :3])

        # estimated pose (timestamp, x, y, z, qw, qx, qy, qz)
        keyframe_pose = np.zeros(8)
        keyframe_pose[0] = timestamp
        keyframe_pose[1:4] = keyframe_pose_matrix[:3, 3]
        keyframe_pose[4:] = r.as_quat()[[3, 0, 1, 2]]
    return keyframe_pose
