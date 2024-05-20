"""
Author:      Dongmyeong Lee
Date:        May 16, 2024
Description: Get global 3D ellipsoid by accumulating 3D annotations of CODa dataset.
"""

import os
import argparse
import pathlib
from natsort import natsorted
import json
from tqdm import tqdm
import time
import warnings
from multiprocessing import Pool

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN
from scipy.optimize import least_squares

from utils.lie_math import xyz_quat_to_matrix, xyz_rpy_to_matrix, average_rpy
from utils.geometry import transform_3d_bbox, get_3d_bbox_planes, transform_plane
from utils.o3d_visualization import (
    O3dVisualizer,
    create_o3d_3d_bbox,
    create_o3d_ellipsoid,
)

try:
    import open3d as o3d

    OPEN3D_AVAIL = True
except ImportError:
    OPEN3D_AVAIL = False
    warnings.warn("Open3D is not installed. Visualization is disabled.")

# fmt: off
CLASSES = {
    "Tree":               {"id": 0, "color": (0, 1.0, 0)},
    "Pole":               {"id": 1, "color": (0, 0, 1.0)},
    "Bollard":            {"id": 2, "color": (1.0, 0, 0)},
    "Informational_Sign": {"id": 3, "color": (1.0, 1.0, 0)},
    "Traffic_Sign":       {"id": 4, "color": (1.0, 0, 1.0)},
    "Trash_Can":          {"id": 5, "color": (0, 1.0, 1.0)},
    "Fire_Hydrant":       {"id": 6, "color": (0.5, 0.5, 0.5)},
    "Emergency_Phone":    {"id": 7, "color": (0.5, 0.5, 0.5)},
    "Railing":            {"id": 8, "color": (0.5, 0.5, 0.5)},
    "Fence":              {"id": 9, "color": (0.5, 0.5, 0.5)},
    "Bike_Rack":          {"id": 10, "color": (0.5, 0.5, 0.5)},
    "Floor_Sign":         {"id": 11, "color": (0.5, 0.5, 0.5)},
    "Traffic_Arm":        {"id": 12, "color": (0.5, 0.5, 0.5)},
    "Traffic_Light":      {"id": 13, "color": (0.5, 0.5, 0.5)},
    "Table":              {"id": 14, "color": (0.5, 0.5, 0.5)},
    "Chair":              {"id": 15, "color": (0.5, 0.5, 0.5)},
    "Door":               {"id": 16, "color": (0.5, 0.5, 0.5)},
    "Wall_Sign":          {"id": 17, "color": (0.5, 0.5, 0.5)},
    "Bench":              {"id": 18, "color": (0.5, 0.5, 0.5)},
    "Couch":              {"id": 19, "color": (0.5, 0.5, 0.5)},
    "Parking_Kiosk":      {"id": 20, "color": (0.5, 0.5, 0.5)},
    "Mailbox":            {"id": 21, "color": (0.5, 0.5, 0.5)},
    "Water_Fountain":     {"id": 22, "color": (0.5, 0.5, 0.5)},
    "ATM":                {"id": 23, "color": (0.5, 0.5, 0.5)},
    "Fire_Alarm":         {"id": 24, "color": (0.5, 0.5, 0.5)},
}
# fmt: on


def distance_to_plane(plane, x):
    """
    Distance from the ellipsoid to the plane using Lagrange multipliers
    https://math.stackexchange.com/questions/1108761/finding-the-distance-from-ellipsoid-to-plane

    distance = min (|D +- sqrt(A^2*a^2 + B^2*b^2 + C^2*c^2)|)

    Compute distance in the ellipsoid frame
    a, b, c: semi-axes of the ellipsoid
    Ax + By + Cz + D = 0: transformed plane
    """
    Twe = xyz_rpy_to_matrix(x[[0, 1, 2, 6, 7, 8]])
    Tew = np.linalg.inv(Twe)
    transformed_plane = transform_plane(plane, Tew)
    A, B, C, D = transformed_plane
    a, b, c = x[3:6]

    r = np.sqrt(A**2 * a**2 + B**2 * b**2 + C**2 * c**2)
    distance = min(abs(D + r), abs(D - r))

    return distance


def fit_ellipsoid(bboxes):
    """
    Fit an ellipsoid to a cluster of 3D bounding boxes
    that minimizes the distance to planes of bounding boxes
    """

    # Initial guess
    cX, cY, cZ, a, b, c = np.median(bboxes[:, :6], axis=0)
    r, p, y = average_rpy(bboxes[:, 6:9])
    x0 = np.array([cX, cY, cZ, a, b, c, r, p, y])

    # Objective function
    def fun(x):
        residuals = []
        for bbox in bboxes:
            planes = get_3d_bbox_planes(bbox)
            for plane in planes:
                distance = distance_to_plane(plane, x)
                residuals.append(distance)
        return residuals

    # Optimization
    result = least_squares(fun, x0, loss="soft_l1", f_scale=10, max_nfev=8, verbose=2)

    return result.x


def cluster_fit_ellipsoid(bboxes, threshold=1.0, min_samples=2, visualize=False):
    """
    Clustering and fitting 3D bounding boxes

    Args:
        bboxes: list of 3D bounding boxes (cX, cY, cZ, l, w, h, r, p, y)
        threshold: DBSCAN threshold
        min_samples: DBSCAN min_samples

    Returns:
        fitted_ellipsoids: list of fitted ellipsoids (cX, cY, cZ, a, b, c, r, p, y)
                           inscribed ellipsoid of the 3D bounding boxes
    """
    if len(bboxes) == 0:
        return []

    vis = O3dVisualizer() if visualize else None

    # Clustering
    centroids = np.array([bbox[:3] for bbox in bboxes])
    clustering = DBSCAN(eps=threshold, min_samples=1).fit(centroids)
    labels = clustering.labels_
    unique_labels = np.unique(labels)

    # Fitting
    fitted_ellipsoids = []
    for unique_label in tqdm(unique_labels):
        indices = np.where(labels == unique_label)[0]

        if len(indices) < min_samples:
            continue

        cluster_bboxes = np.array([bboxes[i] for i in indices])

        # Fit an ellipsoid
        ellipsoid = fit_ellipsoid(cluster_bboxes)

        # Visualize the fitted ellipsoid
        if vis:
            for bbox in cluster_bboxes:
                obb = create_o3d_3d_bbox(bbox, (1.0, 1.0, 1.0))
                vis.add_geometry(obb)
            ell = create_o3d_ellipsoid(ellipsoid, (1.0, 1.0, 0.0))
            vis.add_geometry(ell)
            vis.update()

        fitted_result = {
            "id": int(unique_label),
            "ellipsoid": ellipsoid,
            "indices": indices,
        }
        fitted_ellipsoids.append(fitted_result)

    if vis:
        vis.run()
        vis.close()

    return fitted_ellipsoids


def load_3d_annotations(pose, annotation_file):
    """
    Load 3D bounding box in the global frame

    Args:
        pose: (7, ) array of [x, y, z, qw, qx, qy, qz]
        annotation_file: The annotation file (.json file) (bbox in the LiDAR frame)

    Returns:
        bboxes: dict of 3D bounding boxes in the global frame
                {class_name: [bbox1, bbox2, ...]}
    """
    assert os.path.exists(annotation_file)

    Twl = xyz_quat_to_matrix(pose)
    bboxes = {class_name: [] for class_name in CLASSES}

    for instance in json.load(open(annotation_file))["3dbbox"]:
        class_name = instance["classId"].replace(" ", "_")
        if class_name in CLASSES:
            bbox = np.array(
                [
                    instance[attr]
                    for attr in ["cX", "cY", "cZ", "l", "w", "h", "r", "p", "y"]
                ]
            )
            global_bbox = transform_3d_bbox(bbox, Twl)
            bboxes[class_name].append(global_bbox)

    return bboxes


def main(args):
    global_bboxes = {class_name: [] for class_name in CLASSES}
    fitted_bboxes = {class_name: [] for class_name in CLASSES}

    # Accumulate 3D bounding boxes
    for seq in args.sequences:
        pose_file = args.pose_dir / f"{seq}.txt"
        pose_np = np.loadtxt(pose_file).reshape(-1, 8)
        timestamp_file = args.timestamps_dir / f"{seq}.txt"
        timestamps = np.loadtxt(timestamp_file)
        assert len(pose_np) == len(timestamps), f"{len(pose_np)} != {len(timestamps)}"

        for frame in tqdm(range(len(pose_np)), desc=f"Sequence {seq}"):
            os1_3d_bbox_file = (
                args.os1_3d_bbox_dir / f"{seq}" / f"3d_bbox_os1_{seq}_{frame}.json"
            )
            if not os.path.exists(os1_3d_bbox_file):
                continue

            # bboxes of the frame
            bboxes = load_3d_annotations(pose_np[frame][1:], os1_3d_bbox_file)

            for class_name in CLASSES:
                global_bboxes[class_name].extend(bboxes[class_name])

    # Get fitted 3D bounding boxes
    for class_name in CLASSES:
        print(f"Clustering and fitting {class_name}...")
        fitted_bboxes[class_name] = cluster_fit_ellipsoid(
            global_bboxes[class_name], visualize=args.visualize
        )

        bboxes_json = {
            "class": class_name,
            "instances": [
                {
                    "id": fitted_ellipse["id"],
                    **dict(
                        zip(
                            ["cX", "cY", "cZ", "a", "b", "c", "r", "p", "y"],
                            fitted_ellipse["ellipsoid"],
                        )
                    ),
                    "3d_bboxes": [
                        dict(
                            zip(
                                ["cX", "cY", "cZ", "l", "w", "h", "r", "p", "y"],
                                global_bboxes[class_name][idx],
                            )
                        )
                        for idx in fitted_ellipse["indices"]
                    ],
                }
                for fitted_ellipse in fitted_bboxes[class_name]
            ],
        }

        # Save the results
        out_debug_file = args.global_ellipsoid_dir / f"{class_name}_debug.json"
        with open(out_debug_file, "w") as f:
            json.dump(bboxes_json, f, indent=4)

        for instances in bboxes_json["instances"]:
            instances.pop("3d_bboxes")

        out_file = args.global_ellipsoid_dir / f"{class_name}.json"
        with open(out_file, "w") as f:
            json.dump(bboxes_json, f, indent=4)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data/CODa")
    parser.add_argument(
        "--sequences",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22],
        help="The sequences to process",
    )
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    args.visualize = args.visualize and OPEN3D_AVAIL

    args.dataset_dir = pathlib.Path(args.dataset_dir)
    args.pose_dir = args.dataset_dir / "poses_bkp/dense_global"
    args.timestamps_dir = args.dataset_dir / "timestamps"
    args.os1_3d_bbox_dir = args.dataset_dir / "3d_bbox" / "os1"

    args.global_ellipsoid_dir = args.dataset_dir / "annotations" / "global_ellipsoids"
    os.makedirs(args.global_ellipsoid_dir, exist_ok=True)

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
