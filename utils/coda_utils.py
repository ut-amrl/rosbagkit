import yaml
import numpy as np

from typing import Dict
from scipy.spatial.transform import Rotation as R


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
        camera_params: Dictionary containing the camera parameters.
            - K: (3, 3) intrinsic matrix
            - img_size: (2,) image size
            - D: (5,) distortion coefficients
    """
    with open(intrinsic_file, "r") as f:
        params = yaml.safe_load(f)
        matrix_params = params["camera_matrix"]

        intrinsic_matrix = np.array(matrix_params["data"]).reshape(
            matrix_params["rows"], matrix_params["cols"]
        )
        image_size = np.array([params["image_width"], params["image_height"]])
        distortion_coeffs = np.array(params["distortion_coefficients"]["data"])
    return {"K": intrinsic_matrix, "img_size": image_size, "D": distortion_coeffs}
