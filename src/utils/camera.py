import yaml
import numpy as np

from typing import Dict
from scipy.spatial.transform import Rotation as R


def load_extrinsics(extrinsic_file: str) -> np.ndarray:
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


def load_cam_params(intrinsic_file: str) -> Dict[str, np.ndarray]:
    """
    Load camera parameters from a yaml file.

    Args:
        intrinsic_file: Path to the yaml file containing the camera parameters.
    Returns:
        camera_params: Dictionary containing the camera parameters.
            - img_size: (2,) image size (height, width)
            - K: (3, 3) intrinsic matrix
            - D: (4>=,) distortion coefficients
            - R: (3, 3) rectification matrix
            - P: (3, 4) projection matrix
    """
    with open(intrinsic_file, "r") as f:
        params = yaml.safe_load(f)

        if "image_width" in params.keys() and "image_height" in params.keys():
            image_size = np.array([params["image_height"], params["image_width"]])
        elif "width" in params.keys() and "height" in params.keys():
            image_size = np.array([params["height"], params["width"]])

        intrinsic_matrix = np.array(params["camera_matrix"]["data"]).reshape(
            params["camera_matrix"]["rows"],
            params["camera_matrix"]["cols"],
        )  # (3, 3)

        distortion_coeffs = np.array(params["distortion_coefficients"]["data"])  # (5,)

        rectification_matrix = np.array(params["rectification_matrix"]["data"]).reshape(
            params["rectification_matrix"]["rows"],
            params["rectification_matrix"]["cols"],
        )  # (3, 3)

        projection_matrix = np.array(params["projection_matrix"]["data"]).reshape(
            params["projection_matrix"]["rows"],
            params["projection_matrix"]["cols"],
        )  # (3, 4)

    return {
        "img_size": image_size,
        "K": intrinsic_matrix,
        "D": distortion_coeffs,
        "R": rectification_matrix,
        "P": projection_matrix,
    }
