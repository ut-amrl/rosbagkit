"""
Author:      Dongmyeong Lee (domlee[at]utexas[dot]edu)
Date:        Feb 20, 2024
Description: Get annotations of instances in 2D image from 3D Bounding Box
"""

import os
import pathlib
import argparse
import json
import jsbeautifier
from collections import defaultdict
from tqdm import tqdm
from typing import Optional
import warnings

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

import torch

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
    transform_3d_bbox,
    filter_points_inside_bbox_3d,
)
from utils.image import compute_overlap, ratio_within_image
from utils.ros_viz_utils import (
    create_3d_bbox_marker,
    create_filled_bbox_3d_marker,
    clear_marker_array,
)
from utils.ros_utils import wait_for_subscribers
from utils.math_utils import average_rpy, solve_linear_quadratic
from utils.coda_utils import load_extrinsic_matrix, load_camera_params

DEBUG = False

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
}
SEQ_WEATHER = {
    0: "cloudy",
    1: "cloudy",
    2: "dark",
    3: "sunny",
    4: "dark",
    5: "dark",
    6: "sunny",
    7: "sunny",
    8: "cloudy",
    9: "cloudy",
    10: "cloudy",
    11: "sunny",
    12: "cloudy",
    13: "rainy",
    14: "dark",
    15: "rainy",
    16: "rainy",
    17: "sunny",
    18: "sunny",
    19: "sunny",
    20: "sunny",
    21: "cloudy",
    22: "sunny",
}
# fmt: on

# Radius for KDTree Query
RADIUS = 50.0

# SAM Predictor (Segment Anything) global variable
SAM_PREDICTOR = None


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
        # default=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22],
        default=[0],
        help="Sequence ID",
    )
    parser.add_argument(
        "--cams",
        nargs="+",
        type=str,
        default=["cam0", "cam1"],
        help="Camera Names",
    )

    parser.add_argument(
        "--sam", action="store_true", help="Run with SAM (Segment Anything)"
    )
    parser.add_argument(
        "-m",
        "--sam_model",
        type=str,
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM model name",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/home/dongmyeong/Projects/others/segment-anything/weights/sam_vit_h_4b8939.pth",
        help="Path to the checkpoint of SAM model",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize Output",
    )

    parser.add_argument(
        "--ros",
        action="store_true",
        help="Run with ROS",
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
    args = parser.parse_args()

    # Directories
    args.dataset_dir = pathlib.Path(args.dataset)
    args.pose_dir = args.dataset_dir / "correct"
    args.timestamp_dir = args.dataset_dir / "timestamps"
    args.pointcloud_dir = args.dataset_dir / "3d_comp_bkp" / "os1" / "0"
    args.image_dirs = {cam: args.dataset_dir / "2d_raw" / cam for cam in args.cams}
    args.calibration_dir = args.dataset_dir / "calibrations"
    args.global_bbox_dir = args.dataset_dir / "3d_bbox" / "global"
    args.out_dir = args.dataset_dir / "annotations"
    return args


def load_SAM(args):
    """Load Segment Anything Model"""
    from segment_anything import sam_model_registry, SamPredictor

    global SAM_PREDICTOR
    sam = sam_model_registry[args.sam_model](checkpoint=args.ckpt).to("cuda")
    SAM_PREDICTOR = SamPredictor(sam)
    print("SAM Model Loaded")


def load_kdtree_3d_bbox(args) -> dict:
    """Load 3D Bounding Box and Build KDTree"""
    print("Loading Global 3D Bounding Box and Build KDTree...")
    bboxes_3d = {class_name: [] for class_name in CLASSES}
    for class_name in CLASSES:
        bbox_3d_file = args.global_bbox_dir / f"{class_name}.json"
        if not os.path.exists(bbox_3d_file):
            warnings.warn(f"{bbox_3d_file} does not exist")
            continue

        bbox_3d_json = json.load(open(bbox_3d_file, "r"))
        for idx, instance in enumerate(bbox_3d_json["instances"]):
            instance_id = instance["id"]
            instance_bbox = np.array(
                [
                    instance[attr]
                    for attr in ["cX", "cY", "cZ", "l", "w", "h", "r", "p", "y"]
                ]
            )
            bboxes_3d[class_name].append((instance_id, instance_bbox))

    # Build KDTree for each class with 3D Bounding Box Centroids
    bboxes_kdtree = {
        class_name: KDTree([bbox[:3] for _, bbox in instances])
        for class_name, instances in bboxes_3d.items()
    }

    # Visualize 3D Bounding Box
    if args.ros:
        # 3D Bounding Box Publisher for each class
        object_pubs = {
            class_name: rospy.Publisher(
                f"global_3dbbox/{class_name}", MarkerArray, queue_size=1
            )
            for class_name in CLASSES
        }
        wait_for_subscribers(list(object_pubs.values()))

        for class_name in object_pubs.keys():
            marker_array = MarkerArray()
            for bbox in bboxes_3d[class_name]:
                bbox_marker = create_3d_bbox_marker(
                    bbox_3d=bbox[1],
                    frame_id="map",
                    marker_id=bbox[0],
                    namespace=class_name,
                    color=CLASSES[class_name]["color"],
                )
                marker_array.markers.append(bbox_marker)
            clear_marker_array(object_pubs[class_name])
            object_pubs[class_name].publish(marker_array)

    return bboxes_kdtree, bboxes_3d


def refine_bbox_2d(
    image_file: str,
    drawn_instances: list[list],
):
    """
    Refine 2D Bounding Box with SAM

    Args:
        image_file: Image File Path
        drawn_instances: list of (class_name, instance_id, bbox_2d)

    Returns:
        refined_instances: list of (class_name, instance_id, bbox_2d)
        segmentations: list of contours
    """
    img = cv2.imread(image_file)
    SAM_PREDICTOR.set_image(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    boxes = torch.tensor(
        [bbox for _, _, bbox in drawn_instances], device=SAM_PREDICTOR.device
    )
    boxes = SAM_PREDICTOR.transform.apply_boxes_torch(boxes, img.shape[:2])
    masks, iou_preds, _ = SAM_PREDICTOR.predict_torch(
        point_coords=None, point_labels=None, boxes=boxes, multimask_output=False
    )

    refined_instances = []
    segmentations = []

    used_masks = np.zeros_like(masks[0].cpu().numpy().squeeze())
    for mask, iou_pred, instance in zip(masks, iou_preds, drawn_instances):
        # Filter out low quality masks
        iou_pred = iou_pred.cpu().numpy().squeeze()
        if iou_pred < 0.5:
            if DEBUG:
                print(f"Low quality mask: {iou_pred}")
            continue

        mask = mask.cpu().numpy().squeeze().astype(np.uint8)
        # Filter out small masks
        if mask.sum() < 1000:
            if DEBUG:
                print(f"Small mask: {mask.sum()}")
            continue

        # Draw Contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) < 1:
            continue

        # Get approximated contour
        approxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:
                continue
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            approxes.append(approx)
        if len(approxes) < 1:
            continue

        # Get new bbox
        old_bbox = instance[2]
        x, y, w, h = cv2.boundingRect(np.concatenate(approxes, axis=0))
        new_bbox = [x, y, x + w, y + h]
        new_area = w * h
        old_area = (old_bbox[2] - old_bbox[0]) * (old_bbox[3] - old_bbox[1])
        if new_area < old_area * 0.5:
            continue

        # Update bbox and segmentation
        instance[2] = list(map(int, new_bbox))
        refined_instances.append(instance)
        segmentations.append([c.ravel().tolist() for c in approxes])

    return refined_instances, segmentations


def process_bboxes_frame(
    image_file: str,
    instances_frame: list[list],
    output_file: Optional[str] = None,
    info: dict = None,
    args: argparse.Namespace = None,
):
    """
    Process Bounding Box in a frame

    Args:
        image_file: Image File Path
        instances_frame: instances info in a frame
            - list of (class_name, instance_id, bbox_2d, bbox_3d_lidar, points_in_bbox)
        output_file: (Optional) Output File Path
        info: (Optional) Information of the frame
    """
    # Sort Bounding Box by Distance
    instances_frame.sort(key=lambda x: np.sqrt(x[3][0] ** 2 + x[3][1] ** 2))
    if DEBUG:
        print("\nprocessing bboxes in a frame...")
        for instance in instances_frame:
            print(instance[:3])

    drawn_instances = []
    # get bounding box from closest to farthest
    for instance in instances_frame:
        class_name, instance_id, bbox_2d, _, _ = instance
        is_occluded = any(
            compute_overlap(bbox_2d, drawn_bbox) > 0.5
            for _, _, drawn_bbox in drawn_instances
        )
        bbox_area = (bbox_2d[2] - bbox_2d[0]) * (bbox_2d[3] - bbox_2d[1])
        if not is_occluded and bbox_area > 2000:
            bbox_2d = list(map(int, bbox_2d))
            drawn_instances.append([class_name, instance_id, bbox_2d])

    if len(drawn_instances) == 0:
        return

    # Refine Bounding Box with SAM
    segmentations = []
    if SAM_PREDICTOR is not None:
        drawn_instances, segmentations = refine_bbox_2d(image_file, drawn_instances)
        if DEBUG:
            print("Refined drawn_instances", drawn_instances, segmentations)

    # only save annotation if there is at least one instance
    if len(drawn_instances) == 0:
        return

    # Save Annotation to File
    frame_annotation = {"info": info, "instances": []}
    for class_name, instance_id, bbox_2d in drawn_instances:
        frame_annotation["instances"].append(
            {
                "class": class_name,
                "id": instance_id,
                "bbox": bbox_2d,
            }
        )
    for annotation, segment in zip(frame_annotation["instances"], segmentations):
        annotation["segmentation"] = segment

    if DEBUG:
        print(f"frame_annotation:\n {frame_annotation}\n")

    if output_file is not None:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        opts = jsbeautifier.default_options()
        opts.indent_size = 2
        with open(output_file, "w") as f:
            f.write(jsbeautifier.beautify(json.dumps(frame_annotation), opts))

    # Visualize 2D Bounding Box
    if args.visualize:
        img = cv2.imread(image_file)
        for class_name, instance_id, bbox_2d in drawn_instances:
            try:
                color = CLASSES[class_name]["color"]
                cv2.rectangle(
                    img,
                    (bbox_2d[0], bbox_2d[1]),
                    (bbox_2d[2], bbox_2d[3]),
                    (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)),
                    2,
                )
                cv2.putText(
                    img,
                    f"{class_name} {instance_id}",
                    (bbox_2d[0], bbox_2d[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
            except Exception as e:
                print(e)
                print(
                    "class_name, instance_id, bbox_2d", class_name, instance_id, bbox_2d
                )

        for segment in segmentations:
            for contour in segment:
                cv2.drawContours(
                    img, [np.array(contour).reshape(-1, 1, 2)], -1, (0, 255, 0), 2
                )
        cv2.imshow("2D Bounding Box", img)
        cv2.waitKey(0 if DEBUG else 1)


def get_bbox_2d(
    bbox_3d: np.ndarray,
    pointcloud: np.ndarray,
    extrinsic: np.ndarray,
    cam_params: dict,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Get 2D Bounding Box from 3D Bounding Box by checking points existence in 3D Bounding
    Box and projecting ellipsoid onto 2D Image Plane

    Args:
        bbox_3d: (9,) 3D Bounding Box in LiDAR Frame (cx, cy, cz, l, w, h, r, p, y)
        pointcloud: (N, 3+) Pointcloud in LiDAR Frame (x, y, z, intensity, ...)
        extrinsic: (4, 4) Camera Extrinsic Matrix (LiDAR Frame -> Camera Frame)
        cam_params: Camera Parameters (Intrinsic Matrix, Distortion Coefficients, ...)
    Returns:
        bbox_2d: (4,) 2D Bounding Box in Image Frame (x1, y1, x2, y2)
        points_in_bbox: (M, 3+) Points in 3D Bounding Box in LiDAR Frame
    """
    assert bbox_3d.shape == (9,), "Invalid 3D Bounding Box Shape"
    assert pointcloud.shape[1] >= 3, "Invalid Pointcloud Shape"
    assert extrinsic.shape == (4, 4), "Invalid Camera Extrinsic Matrix Shape"
    assert cam_params["K"].shape == (3, 3), "Invalid Camera Intrinsic Matrix Shape"
    assert cam_params["img_size"].shape == (2,), "Invalid Image Size Shape"

    w, h = cam_params["img_size"]

    # Check whether 3D Bounding Box is in front of the camera
    centroid_cam = np.dot(extrinsic, np.append(bbox_3d[:3], 1.0))[:3]
    if centroid_cam[2] < 0.1:  # 3D Bounding Box is behind the camera
        if DEBUG:
            print("centroid_cam: ", centroid_cam)
        return None, None

    # Project centroid to 2D Image Plane
    centroid_img, _ = cv2.projectPoints(
        centroid_cam, np.eye(3), np.zeros(3), cam_params["K"], cam_params["D"]
    )
    if (
        centroid_img[0, 0, 0] < 0
        or centroid_img[0, 0, 0] >= w
        or centroid_img[0, 0, 1] < 0
        or centroid_img[0, 0, 1] >= h
    ):
        if DEBUG:
            print("centroid_img: ", centroid_img)
        return None, None

    # Count Points in 3D Bounding Box
    bbox_3d_margin = bbox_3d.copy()
    # bbox_3d_margin[3:5] += 1.0  # Add margin to length and width
    bbox_3d_margin[5] -= 0.2  # reduce height (to avoid the ground)
    points_in_bbox = filter_points_inside_bbox_3d(pointcloud, bbox_3d_margin)

    # bbox is not visible
    if len(points_in_bbox) < 3:
        if DEBUG:
            print(f"points_in_bbox: {len(points_in_bbox)}")
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
    P = cam_params["K"] @ extrinsic[:3]  # Projection Matrix (3, 4)
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


def main():
    args = get_args()
    if args.sam:
        load_SAM(args)

    if args.ros:
        rospy.init_node("get_annotations_2d")

    # Load 3D Bounding Box and Build KDTree
    bboxes_kdtree, bboxes_3d = load_kdtree_3d_bbox(args)
    print("3D Bounding Box and KDTree Loaded")

    # Main Loop
    for seq in args.sequences:
        print(f"* Processing Sequence {seq}...")
        lidar_pose_np = np.loadtxt(args.pose_dir / f"{seq}.txt")[:, :8]
        timestamp_np = np.loadtxt(args.timestamp_dir / f"{seq}.txt")

        # Calibration
        os1_to_cam_extrinsics = {
            cam: load_extrinsic_matrix(
                args.calibration_dir / str(seq) / f"calib_os1_to_{cam}.yaml"
            )
            for cam in args.cams
        }
        t_cl = {cam: np.linalg.inv(os1_to_cam_extrinsics[cam]) for cam in args.cams}
        cam_params = {
            cam: load_camera_params(
                args.calibration_dir / str(seq) / f"calib_{cam}_intrinsics.yaml"
            )
            for cam in args.cams
        }

        # Main Loop
        for lidar_pose in tqdm(lidar_pose_np[::50], total=len(lidar_pose_np)):
            # Get Pose
            frame = np.searchsorted(timestamp_np, lidar_pose[0], side="left")

            T_lg = np.eye(4)
            T_lg[:3, 3] = lidar_pose[1:4]
            T_lg[:3, :3] = R.from_quat(lidar_pose[[5, 6, 7, 4]]).as_matrix()
            T_gl = np.linalg.inv(T_lg)

            # Query 3D Bounding Box
            queried_indices = {}
            for class_name, class_bboxes_kdtree in bboxes_kdtree.items():
                indices = class_bboxes_kdtree.query_ball_point(
                    lidar_pose[1:4], r=RADIUS
                )
                queried_indices[class_name] = indices

            if all(len(indices) == 0 for indices in queried_indices.values()):
                continue

            # Pointcloud & Image files
            pc_file = args.pointcloud_dir / str(seq) / f"3d_comp_os1_{seq}_{frame}.bin"
            if not pc_file.exists():
                continue
            image_files = {
                cam: args.image_dirs[cam] / str(seq) / f"2d_raw_{cam}_{seq}_{frame}.jpg"
                for cam in args.cams
            }
            # Load Pointcloud & Images
            pointcloud = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 4)

            # Project queried 3D Bounding Box to 2D Bounding Box
            for cam in args.cams:
                instances_frame = []
                for class_name, indices in queried_indices.items():
                    for idx in indices:
                        instance_id, instance_bbox = bboxes_3d[class_name][idx]
                        bbox_3d_lidar = transform_3d_bbox(instance_bbox, T_gl)

                        # Get 2D Bounding Box by projecting 3D Bounding Box
                        bbox_2d, points_in_bbox = get_bbox_2d(
                            bbox_3d_lidar,
                            pointcloud,
                            os1_to_cam_extrinsics[cam],
                            cam_params[cam],
                        )
                        if bbox_2d is None:
                            continue

                        instances_frame.append(
                            [
                                class_name,
                                instance_id,
                                bbox_2d,
                                bbox_3d_lidar,
                                points_in_bbox,
                            ]
                        )
                if len(instances_frame) == 0:
                    continue

                T_cg = T_lg @ t_cl[cam]
                cam_pose = np.zeros(7)
                cam_pose[:3] = T_cg[:3, 3]
                cam_pose[3:] = np.roll(R.from_matrix(T_cg[:3, :3]).as_quat(), 1)
                info = {
                    "image_file": str(image_files[cam].stem),
                    "img_size": cam_params[cam]["img_size"].tolist(),
                    "camera": cam,
                    "sequence": seq,
                    "frame": int(frame),
                    "weather_condition": SEQ_WEATHER[seq],
                    "pose": [round(p, 6) for p in cam_pose],
                }

                # Post-Processing 2D Bounding Box
                output_file = str(args.out_dir / cam / str(seq) / f"{frame}.json")
                process_bboxes_frame(
                    str(image_files[cam]), instances_frame, output_file, info, args
                )


def publish_bbox_pc(pointcloud, bbox, T_l_g):
    """publish 3D Bounding Box and Pointcloud"""


if __name__ == "__main__":
    main()
