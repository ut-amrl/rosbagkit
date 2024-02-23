"""
Author:      Dongmyeong Lee (domlee[at]utexas[dot]edu)
Date:        Nov 23, 2023
Description: Get insatance 2D bounding box from 3D bounding box in global frame
"""
import os
import pathlib
import argparse
import json
from tqdm import tqdm
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

import rospy
import cv2
from cv_bridge import CvBridge
import tf2_ros

from sensor_msgs.msg import PointCloud2, Image
from visualization_msgs.msg import Marker, MarkerArray
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry, Path

from utils.msg_converter import (
    np_to_pointcloud2,
    pose_stamped_from_xyz_quat,
    odometry_from_xyz_quat,
    tf_msg_from_quat,
)
from utils.geometry import (
    transform_bbox_3d,
    project_points_3d_to_2d,
    filter_points_inside_bbox_3d,
)
from utils.image_utils import compute_overlap, ratio_within_image
from utils.ros_viz_utils import (
    create_bbox_3d_marker,
    create_filled_bbox_3d_marker,
    clear_marker_array,
)
from utils.ros_utils import wait_for_subscribers
from utils.math_utils import average_rpy, solve_linear_quadratic
from utils.coda_utils import load_extrinsic_matrix, load_camera_params

# Frames
global_frame = "map"
lidar_frame = "base_link"
cam_list = ["cam0", "cam1"]

# Classes to be detected
CLASSES = {
    "Tree": {"id": 0, "color": (0, 255, 0)},
    "Pole": {"id": 1, "color": (0, 0, 255)},
    "Bollard": {"id": 2, "color": (255, 0, 0)},
}

# Radius for KDTree Query
RADIUS = 25.0


def get_args():
    parser = argparse.ArgumentParser(description="Get instance 2D bounding box")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="/home/dongmyeong/Projects/AMRL/CODa",
        help="Path to the dataset",
    )
    parser.add_argument(
        "-s",
        "--sequences",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21],
        help="Sequence ID",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0,
        help="Publishing rate (0: no delay)",
    )
    parser.add_argument(
        "--map_topic",
        type=str,
        default="/global_map",
        help="Global map topic name",
    )
    parser.add_argument(
        "--pointcloud_topic",
        type=str,
        default="/points",
        help="Pointcloud topic name",
    )
    parser.add_argument(
        "--image_topic",
        type=str,
        default="/image",
        help="Image topic name",
    )
    return parser.parse_args()


def get_bbox_2d(
    bbox_3d: np.ndarray,
    pointcloud: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Get 2D Bounding Box from 3D Bounding Box by checking points existence in 3D Bounding
    Box and projecting ellipsoid onto 2D Image Plane

    Args:
        bbox_3d: (9,) 3D Bounding Box in LiDAR Frame (cx, cy, cz, l, w, h, r, p, y)
        pointcloud: (N, 3+) Pointcloud in LiDAR Frame (x, y, z, intensity, ...)
        extrinsic: (4, 4) Camera Extrinsic Matrix (LiDAR Frame -> Camera Frame)
        intrinsic: (3, 3) Camera Intrinsic Matrix (Camera Frame -> Image Frame)
        image_size: (width, height) Image Size
    Returns:
        bbox_2d: (4,) 2D Bounding Box in Image Frame (x1, y1, x2, y2)
        points_in_bbox: (M, 3+) Points in 3D Bounding Box in LiDAR Frame
    """
    assert bbox_3d.shape == (9,), "Invalid 3D Bounding Box Shape"
    assert pointcloud.shape[1] >= 3, "Invalid Pointcloud Shape"
    assert extrinsic.shape == (4, 4), "Invalid Camera Extrinsic Matrix Shape"
    assert intrinsic.shape == (3, 3), "Invalid Camera Intrinsic Matrix Shape"
    assert len(image_size) == 2, "Invalid Image Size"

    w, h = image_size

    # Check whether 3D Bounding Box is in front of the camera
    centroid_cam = np.dot(extrinsic, np.append(bbox_3d[:3], 1.0))[:3]
    if centroid_cam[2] < 1e-6:
        return None, None

    # Count Points in 3D Bounding Box
    bbox_3d_margin = bbox_3d.copy()
    bbox_3d_margin[3:5] += 1.0  # Add margin to length and width
    bbox_3d_margin[5] -= 0.2  # Add margin to height
    points_in_bbox = filter_points_inside_bbox_3d(pointcloud, bbox_3d_margin)

    # bbox is not visible
    if len(points_in_bbox) < 30:
        return None, None

    # Modify centroid (cX, cY) to be the average of points in bbox
    centroid = np.mean(points_in_bbox[:, :2], axis=0)
    bbox_3d[:2] = centroid

    # Get Ellipsoid from 3D Bounding Box
    rot = R.from_euler("xyz", bbox_3d[6:9]).as_matrix()
    t = bbox_3d[:3]

    # Adjoint Quadratic Form
    D = np.diag((bbox_3d[3:6] / 2.0) ** 2)
    Q = np.zeros((4, 4))
    Q[:3, :3] = rot @ D @ rot.T - np.outer(t, t)
    Q[:3, 3] = -t
    Q[3, :3] = -t.T
    Q[3, 3] = -1.0

    # Project Ellipsoid to Image Plane
    P = intrinsic @ extrinsic[:3]  # Projection Matrix (3, 4)
    G = P @ Q @ P.T  # Dual Conic Matrix (3, 3)

    # 1. Get Extrema of Ellipsoid
    x_extrema = np.roots([G[2, 2], -2 * G[0, 2], G[0, 0]])
    y_extrema = np.roots([G[2, 2], -2 * G[1, 2], G[1, 1]])
    x_extrema = np.real(x_extrema[np.isreal(x_extrema)])
    y_extrema = np.real(y_extrema[np.isreal(y_extrema)])
    if len(x_extrema) == 0 or len(y_extrema) == 0:
        return None, None

    x_min, x_max = np.min(x_extrema), np.max(x_extrema)
    y_min, y_max = np.min(y_extrema), np.max(y_extrema)

    # Projected ellipsoid is inside the image
    if x_min >= 0 and x_max < w and y_min >= 0 and y_max < h:
        bbox_2d = np.array([x_min, y_min, x_max, y_max])
        return bbox_2d, points_in_bbox

    # compute points of extrema points
    extrema = []
    extrema.append(solve_linear_quadratic(G, np.array([-1, 0, x_min])))
    extrema.append(solve_linear_quadratic(G, np.array([-1, 0, x_max])))
    extrema.append(solve_linear_quadratic(G, np.array([0, -1, y_min])))
    extrema.append(solve_linear_quadratic(G, np.array([0, -1, y_max])))
    extrema = np.vstack(extrema)

    # compute intersection of conic with x=0, x=width, y=0, y=height
    intersections = []
    if x_min < 0:
        intersections.append(solve_linear_quadratic(G, np.array([1, 0, 0])))
    if x_max >= w:
        intersections.append(solve_linear_quadratic(G, np.array([-1, 0, w])))
    if y_min < 0:
        intersections.append(solve_linear_quadratic(G, np.array([0, 1, 0])))
    if y_max >= h:
        intersections.append(solve_linear_quadratic(G, np.array([0, -1, h])))
    intersections = np.vstack(intersections)

    # get points inside the image
    points = np.vstack((extrema, intersections))
    points = points[~np.isnan(points).any(axis=1) & np.isreal(points).all()]
    # points[:, 0] = np.where(np.isclose(points[:, 0], 0, atol=1e-3), 0, points[:, 0])
    # points[:, 0] = np.where(np.isclose(points[:, 0], w, atol=1e-3), w, points[:, 0])
    # points[:, 1] = np.where(np.isclose(points[:, 1], 0, atol=1e-3), 0, points[:, 1])
    # points[:, 1] = np.where(np.isclose(points[:, 1], h, atol=1e-3), h, points[:, 1])
    points[:, 0] = np.clip(points[:, 0], 0, w)
    points[:, 1] = np.clip(points[:, 1], 0, h)

    points = points[
        (points[:, 0] >= 0)
        & (points[:, 0] <= w)
        & (points[:, 1] >= 0)
        & (points[:, 1] <= h)
    ]

    if len(points) < 3:
        return None, None

    bbox_2d = np.array(
        [
            np.min(points[:, 0]),
            np.min(points[:, 1]),
            np.max(points[:, 0]),
            np.max(points[:, 1]),
        ]
    )

    return bbox_2d, points_in_bbox


def process_bboxes_frame(
    img: np.ndarray,
    bbox_frame: list[tuple[str, np.ndarray, np.ndarray]],
    output_file: Optional[str] = None,
) -> np.ndarray:
    """
    Process Bounding Box in a frame

    Args:
        img: (H, W, 3) Image
        bbox_frame: list of tuple(instance_id, bbox_2d, bbox_3d_lidar)
        output_file: (Optional) Output File Path
    """
    # Sort Bounding Box by Distance
    bbox_frame.sort(key=lambda x: np.sqrt(x[2][0] ** 2 + x[2][1] ** 2))

    drawn_bboxes = []
    # Draw Bounding Box
    for instance_id, bbox_2d, _ in bbox_frame:
        if any(
            compute_overlap(bbox_2d, drawn_bbox) > 0.5 for _, drawn_bbox in drawn_bboxes
        ):
            continue

        x1 = np.max((0, bbox_2d[0])).astype(int)
        y1 = np.max((0, bbox_2d[1])).astype(int)
        x2 = np.min((img.shape[1], bbox_2d[2])).astype(int)
        y2 = np.min((img.shape[0], bbox_2d[3])).astype(int)
        bbox_2d = np.array([x1, y1, x2, y2])

        # Draw Bounding Box
        cv2.rectangle(
            img,
            (int(bbox_2d[0]), int(bbox_2d[1])),
            (int(bbox_2d[2]), int(bbox_2d[3])),
            CLASSES[instance_id.split("_")[0]]["color"],
            2,
        )
        cv2.putText(
            img,
            instance_id,
            (int(bbox_2d[0]), int(bbox_2d[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            CLASSES[instance_id.split("_")[0]]["color"],
            2,
        )
        drawn_bboxes.append((instance_id, bbox_2d))

    # Save Bounding Box Information
    if output_file is not None:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            for (
                instance_id,
                bbox_2d,
            ) in drawn_bboxes:
                class_name, instance_id = instance_id.split("_")
                f.write(
                    f"{class_name} {instance_id} {bbox_2d[0]} {bbox_2d[1]} {bbox_2d[2]} {bbox_2d[3]}\n"
                )


def main():
    args = get_args()

    # ROS Initialization
    rospy.set_param("use_sim_time", True)
    rospy.init_node("CODa_instance_2d_bbox_getter", anonymous=True)

    # Data Publishers
    clock_pub = rospy.Publisher("/clock", Clock, queue_size=10)
    map_pub = rospy.Publisher(args.map_topic, PointCloud2, queue_size=1, latch=True)
    pc_pub = rospy.Publisher(args.pointcloud_topic, PointCloud2, queue_size=10)
    pc_in_bbox_pub = rospy.Publisher("/points_in_bbox", PointCloud2, queue_size=10)
    img_pubs = {
        cam: rospy.Publisher(f"/{cam}/image_rect", Image, queue_size=1)
        for cam in cam_list
    }
    # Odometry & Path Publisher
    odom_pub = rospy.Publisher("/odom", Odometry, queue_size=1)
    path_pub = rospy.Publisher("/global_path", Path, queue_size=1)
    # Global Path Marker
    global_path = Path()
    global_path.header.frame_id = global_frame
    # Define TF Broadcaster
    tf_broadcaster = tf2_ros.TransformBroadcaster()

    # 3D Bounding Box Publisher for each class
    object_pubs = {
        class_name: rospy.Publisher(
            f"instance_3dbbox/{class_name}", MarkerArray, queue_size=1, latch=True
        )
        for class_name in CLASSES
    }
    query_object_pubs = {
        class_name: rospy.Publisher(
            f"queried_3dbbox/{class_name}", MarkerArray, queue_size=1
        )
        for class_name in CLASSES
    }

    # Wait for Subscribers
    wait_for_subscribers([*object_pubs.values(), *query_object_pubs.values()])

    # Data Path
    dataset_path = pathlib.Path(args.dataset)

    # Process Global 3D Bounding Box
    bboxes_3d = {class_name: [] for class_name in CLASSES}
    for class_name in CLASSES:
        bbox_3d_file = dataset_path / "3d_bbox" / "global" / f"{class_name}.json"
        if not os.path.exists(bbox_3d_file):
            continue

        marker_array = MarkerArray()

        bbox_3d_json = json.load(open(bbox_3d_file, "r"))
        for idx, bbox_3d in enumerate(bbox_3d_json["3dbbox"]):
            # fmt: off
            bbox = np.array([
                bbox_3d["cX"], bbox_3d["cY"], bbox_3d["cZ"],
                bbox_3d["l"], bbox_3d["w"], bbox_3d["h"],
                bbox_3d["r"], bbox_3d["p"], bbox_3d["y"]
            ])
            # fmt: on
            bbox_marker = create_bbox_3d_marker(
                bbox_3d=bbox,
                frame_id=global_frame,
                marker_id=idx,
                namespace=bbox_3d["instanceId"],
                color=CLASSES[class_name]["color"],
            )
            marker_array.markers.append(bbox_marker)
            # Save 3D Bounding Box
            bboxes_3d[class_name].append((bbox_3d["instanceId"], bbox))

        # Publish Global 3D Bounding Box
        clear_marker_array(object_pubs[class_name])
        object_pubs[class_name].publish(marker_array)

    # Build KDTree for each class with 3D Bounding Box Centroids
    bboxes_kdtree = {
        class_name: KDTree(np.array([bbox[:3] for _, bbox in bboxes_3d[class_name]]))
        for class_name in CLASSES
    }

    # Main Loop
    for seq in args.sequences:
        pose_file = dataset_path / "poses" / "correct" / f"{seq}.txt"
        pose_np = np.loadtxt(pose_file, delimiter=" ")[:, :8]

        timestamp_file = dataset_path / "timestamps" / f"{seq}.txt"
        timestamp_np = np.loadtxt(timestamp_file, delimiter=" ")

        # Data Path for sequence
        pc_root_dir = dataset_path / "3d_comp" / "os1" / str(seq)
        img_root_dirs = {
            cam: dataset_path / "2d_rect" / cam / str(seq) for cam in cam_list
        }

        # Calibration Params
        calib_dir = dataset_path / "calibrations" / str(seq)
        os1_to_cam_extrinsic = {
            cam: load_extrinsic_matrix(calib_dir / f"calib_os1_to_{cam}.yaml")
            for cam in cam_list
        }
        cam_intrinsics = {
            cam: load_camera_params(calib_dir / f"calib_{cam}_intrinsics.yaml")
            for cam in cam_list
        }

        # Main Loop
        for pose in tqdm(pose_np, total=len(pose_np)):
            # Get Pose
            frame = np.searchsorted(timestamp_np, pose[0], side="left")
            ts = rospy.Time.from_sec(pose[0])

            # Publish Clock
            clock_pub.publish(ts)

            # Publish LiDAR Odometry and Path
            odom_msg = odometry_from_xyz_quat(
                pose[1:4], pose[4:8], global_frame, lidar_frame, ts
            )
            odom_pub.publish(odom_msg)

            pose_msg = pose_stamped_from_xyz_quat(
                pose[1:4], pose[4:8], global_frame, ts
            )
            global_path.poses.append(pose_msg)
            path_pub.publish(global_path)

            # Publish TF
            tf_msg = tf_msg_from_quat(
                pose[1:4], pose[4:8], global_frame, lidar_frame, ts
            )
            tf_broadcaster.sendTransform(tf_msg)

            # Transformation Matrix from Global to LiDAR
            H_lg = np.eye(4)
            H_lg[:3, :3] = R.from_quat(pose[[5, 6, 7, 4]]).as_matrix()
            H_lg[:3, 3] = pose[1:4]
            H_gl = np.linalg.inv(H_lg)

            # Data Path for frame
            pc_file = pc_root_dir / f"3d_comp_os1_{seq}_{frame}.bin"
            img_files = {
                cam: img_root_dirs[cam] / f"2d_rect_{cam}_{seq}_{frame}.jpg"
                for cam in cam_list
            }

            # Publish Pointcloud
            pc_np = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 4)
            pc_msg = np_to_pointcloud2(pc_np, "x y z intensity", lidar_frame, ts)
            pc_pub.publish(pc_msg)

            # Query 3D Bounding Box
            queried_indices = {}
            for class_name in CLASSES:
                indices = bboxes_kdtree[class_name].query_ball_point(
                    pose[1:4], r=RADIUS
                )
                queried_indices[class_name] = indices

                # Get Queried 3D Bounding Box
                marker_array = MarkerArray()
                for idx in indices:
                    instance_id, bbox = bboxes_3d[class_name][idx]
                    bbox_marker = create_filled_bbox_3d_marker(
                        bbox_3d=bbox,
                        frame_id=global_frame,
                        marker_id=int(instance_id.split("_")[-1]),
                        namespace=instance_id,
                        color=(0.5, 1.0, 0.5),
                        alpha=0.5,
                    )
                    marker_array.markers.append(bbox_marker)

                # Publish Queried 3D Bounding Box
                clear_marker_array(query_object_pubs[class_name])
                query_object_pubs[class_name].publish(marker_array)

            # Project 3D Bounding Box to 2D Bounding Box
            pc_in_bbox = np.empty((0, 4))
            for cam in cam_list:
                # list of tuple(instance_id, bbox_2d, bbox_3d_lidar)
                bboxes_frame: list[tuple[str, np.ndarray, np.ndarray]] = []
                img = cv2.imread(str(img_files[cam]))
                for class_name in CLASSES:
                    for idx in queried_indices[class_name]:
                        instance_id, bbox_3d = bboxes_3d[class_name][idx]
                        bbox_3d_lidar = transform_bbox_3d(bbox_3d, H_gl)

                        # Get 2D Bounding Box
                        bbox_2d, points_in_bbox = get_bbox_2d(
                            bbox_3d_lidar,
                            pc_np,
                            os1_to_cam_extrinsic[cam],
                            cam_intrinsics[cam]["K"],
                            cam_intrinsics[cam]["img_size"],
                        )

                        if points_in_bbox is None or bbox_2d is None:
                            continue

                        bboxes_frame.append((instance_id, bbox_2d, bbox_3d_lidar))
                        pc_in_bbox = np.vstack((pc_in_bbox, points_in_bbox))

                # Process 2D Bounding Box
                output_file = dataset_path / "2d_bbox" / cam / str(seq) / f"{frame}.txt"
                process_bboxes_frame(img, bboxes_frame, output_file)

                # Publish Processed Image
                img_msg = CvBridge().cv2_to_imgmsg(img, "bgr8")
                img_pubs[cam].publish(img_msg)

            # Publish Pointcloud in Bounding Box
            pc_in_bbox_msg = np_to_pointcloud2(
                pc_in_bbox, "x y z intensity", lidar_frame, ts
            )
            pc_in_bbox_pub.publish(pc_in_bbox_msg)

            # Wait for next frame
            if args.rate > 0:
                time.sleep(1 / args.rate)


if __name__ == "__main__":
    main()
