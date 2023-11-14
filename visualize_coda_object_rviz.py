"""
Author:      Dongmyeong Lee (domlee[at]utexas[dot]edu)
Date:        September 16, 2023
Description: Visualize the data from CODa dataset in RViz and
             project 3D Bounding Box to 2D image.
"""
import os
import pathlib
import argparse
import json
import sys
import termios
import tty
from tqdm import tqdm

import numpy as np
from scipy.spatial.transform import Rotation

import rospy
from cv_bridge import CvBridge
import cv2
import tf2_ros
import tf.transformations as tf_trans

from sensor_msgs.msg import PointCloud2, Image
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path

from helpers.msg_converter import (
    bin_to_pointcloud2,
    np_to_pointcloud2,
    pose_stamped_from_quat,
    pose_stamped_from_matrix,
    tf_msg_from_matrix,
)
from helpers.ros_visualization import (
    clear_marker_array,
    create_3d_bbox_marker,
    create_text_marker,
)
from helpers.geometry import (
    project_bbox_3d_to_2d,
    project_points_3d_to_2d,
    filter_points_inside_3d_bbox,
    get_corners_3d_bbox,
)
from object_mapper import get_2d_bboxes

from datasets.CODa.coda_utils import load_extrinsic_matrix, load_camera_params
from datasets.CODa.constants import *

# from helpers.ros_visualization import publish_3d_bbox


OCCUMULATE_FRAME = 5
DIST_THRESHOLD_PROJECT = 2
DIST_THRESHOLD_ID = 1.0
DISABLE_IOU_THRESHOLD = 0.9


def main(args):
    rospy.set_param("use_sim_time", True)
    rospy.init_node("CODa_publisher")

    # Define Frames
    global_frame = "map"
    lidar_frame = "os1"

    # Define Publishers
    pc_pub = rospy.Publisher("/os1/pointcloud", PointCloud2, queue_size=1)
    bbox_3d_pub = rospy.Publisher("/bbox_3d", MarkerArray, queue_size=1)

    cam_pubs = {
        cam: rospy.Publisher(f"/{cam}/image", Image, queue_size=1)
        for cam in ["cam0", "cam1"]
    }

    # Define Pose Publishers
    lidar_pose_pub = rospy.Publisher("/os1/pose", PoseStamped, queue_size=1)
    path_pub = rospy.Publisher("/global_path", Path, queue_size=1)

    # Object ID Publisher
    object_id_pub = rospy.Publisher("/object_id", MarkerArray, queue_size=1)

    # Define Global Path Marker
    global_path = Path()
    global_path.header.frame_id = global_frame

    # Define TF Broadcaster
    tf_broadcaster = tf2_ros.TransformBroadcaster()

    # Path to the data
    dataset_path = pathlib.Path(args.dataset_path)
    sequence = args.sequence
    pc_root_dir = dataset_path / "3d_comp" / "os1" / sequence
    bbox_3d_root_dir = dataset_path / "3d_pred" / "os1" / sequence
    cam0_root_dir = dataset_path / "2d_rect" / "cam0" / sequence
    cam1_root_dir = dataset_path / "2d_rect" / "cam1" / sequence
    calib_dir = dataset_path / "calibrations" / sequence

    # Output Directory
    bbox_2d_out_dirs = {
        "cam0": dataset_path / "2d_proj" / "cam0" / sequence,
        "cam1": dataset_path / "2d_proj" / "cam1" / sequence,
    }
    for cam, out_dir in bbox_2d_out_dirs.items():
        os.makedirs(out_dir, exist_ok=True)

    # Pose DATA
    pose_file = dataset_path / "poses" / "dense" / f"{sequence}.txt"
    pose_np = np.fromfile(pose_file, sep=" ").reshape(-1, 8)
    lidar_ts_file = dataset_path / "timestamps" / f"{sequence}.txt"

    # Calibration DATA (Extrinsic and Intrinsic)
    os1_to_base_ext_file = calib_dir / "calib_os1_to_base.yaml"
    os1_to_cam_ext_files = {
        "cam0": calib_dir / "calib_os1_to_cam0.yaml",
        "cam1": calib_dir / "calib_os1_to_cam1.yaml",
    }
    cam_intrinsic_files = {
        "cam0": calib_dir / "calib_cam0_intrinsics.yaml",
        "cam1": calib_dir / "calib_cam1_intrinsics.yaml",
    }

    os1_to_base_ext = load_extrinsic_matrix(os1_to_base_ext_file)
    os1_to_cam_extrinsic = {
        cam: load_extrinsic_matrix(ext_file)
        for cam, ext_file in os1_to_cam_ext_files.items()
    }
    cam_intrinsics = {
        cam: load_camera_params(intrinsic_file)
        for cam, intrinsic_file in cam_intrinsic_files.items()
    }

    # ID and global coordinates of Stationary Object
    # {classname: {uID, [centroid in global frame (3,)]}}
    instance_uid_coords = {classname: {} for classname in BBOX_CLASS_VIZ_LIST}

    # Accumulated points of Stationary Object
    # {uID: [points (N, 4) for each frame]}
    accumulated_points = {}

    def process_frame(bbox_3d_json, points, cam_files):
        pass

    # Main Loop
    for sequence in range(22):
        for frame, pose in tqdm(enumerate(pose_np), total=len(pose_np), miniters=1):
            # Get Pose
            ts, x, y, z, qw, qx, qy, qz = pose

            # Find the closest timestamp
            # frame = np.searchsorted(lidar_ts_np, ts, side='left')
            ts = rospy.Time.from_sec(ts)

            # Get LiDAR Pose in global frame in SE(3)
            lidar_pose = np.eye(4)
            lidar_pose[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            lidar_pose[:3, 3] = [x, y, z]

            # Publish LiDAR Pose and Path
            pose_msg = pose_stamped_from_quat(pose[1:], global_frame, ts)
            global_path.poses.append(pose_msg)
            lidar_pose_pub.publish(pose_msg)
            path_pub.publish(global_path)

            # Broadcast TF (map -> os1)
            tf_msg = tf_msg_from_matrix(lidar_pose, global_frame, lidar_frame, ts)
            tf_broadcaster.sendTransform(tf_msg)

            # Get the file path to the data
            pc_file = pc_root_dir / f"3d_comp_os1_{sequence}_{frame}.bin"
            bbox_file = bbox_3d_root_dir / f"3d_bbox_os1_{sequence}_{frame}.json"
            cam_files = {
                "cam0": cam0_root_dir / f"2d_rect_cam0_{sequence}_{frame}.jpg",
                "cam1": cam1_root_dir / f"2d_rect_cam1_{sequence}_{frame}.jpg",
            }

            # Publish Point Cloud
            if os.path.exists(pc_file) and args.visualize_pointcloud:
                pc_msg = bin_to_pointcloud2(pc_file, "xyzi", lidar_frame, ts)
                pc_pub.publish(pc_msg)

            # Skip if the frame is not annotated
            if not os.path.exists(bbox_file):
                if args.visualize_image:
                    # Publish Image
                    for cam, cam_pub in cam_pubs.items():
                        cam_file = cam_files[cam]
                        if not os.path.exists(cam_file):
                            continue
                        image = cv2.imread(str(cam_file), cv2.IMREAD_COLOR)
                        cam_msg = CvBridge().cv2_to_imgmsg(image)
                        cam_msg.header.stamp = ts
                        cam_pub.publish(cam_msg)
                continue

            # Read 3D Bounding Box
            bbox_3d_json = json.load(open(bbox_file, "r"))

            # get the 2D Bounding Box from 3D Bounding Box
            points = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 4)

            # Get Instance information for current frame
            objects_centroid = []
            objects_3d_bbox = []
            accumulated_objects_points = []
            objects_uid = []
            objects_class = []
            for bbox_3d in bbox_3d_json["3dbbox"]:
                # Skip if the object is not stationary
                if bbox_3d["classId"] not in instance_uid_coords:
                    continue

                # Object informations (centroid, bbox, points)
                centroid = np.array([bbox_3d["cX"], bbox_3d["cY"], bbox_3d["cZ"]])
                bbox = (
                    bbox_3d["cX"],
                    bbox_3d["cY"],
                    bbox_3d["cZ"],
                    bbox_3d["l"],
                    bbox_3d["w"],
                    bbox_3d["h"],
                    bbox_3d["r"],
                    bbox_3d["p"],
                    bbox_3d["y"],
                )
                object_points = filter_points_inside_3d_bbox(points[:, :3], bbox)
                objects_centroid.append(centroid)
                objects_3d_bbox.append(bbox)

                # Check if the object is already detected with centroid
                centroid_global = (lidar_pose @ np.array([*centroid, 1]))[:3]
                uid = None
                class_name = bbox_3d["classId"]
                for tmp_uid, tmp_coords in instance_uid_coords[class_name].items():
                    dist = np.linalg.norm(centroid_global - tmp_coords)
                    if dist < DIST_THRESHOLD_ID:
                        uid = tmp_uid
                        break
                # Add new object if not detected
                if not uid:
                    num_obj = len(instance_uid_coords[class_name])
                    uid = f"{''.join(class_name.split())}_{sequence}_{num_obj}"
                    accumulated_points[uid] = []
                objects_uid.append(uid)
                objects_class.append(class_name)

                # Update coordinates of instance in global frame
                instance_uid_coords[class_name][uid] = centroid_global
                # Accumulate point cloud of instance
                object_points_global = lidar_pose @ np.row_stack(
                    (object_points.T, np.ones(object_points.shape[0]))
                )
                accumulated_points[uid].append(object_points_global)
                # keep only the last 5 frames
                if len(accumulated_points[uid]) > OCCUMULATE_FRAME:
                    accumulated_points[uid].pop(0)
                if class_name == "Pedestrian":
                    if len(accumulated_points[uid]) > 1:
                        accumulated_points[uid].pop(0)

                # Accumulate object points in LiDAR frame
                object_points = (
                    np.linalg.inv(lidar_pose) @ np.column_stack(accumulated_points[uid])
                )[:3]
                accumulated_objects_points.append(object_points.T)

            # Project Stationary Object to 2D and get Bounding Box
            for cam, cam_pub in cam_pubs.items():
                os1_to_cam_ext = os1_to_cam_extrinsic[cam]
                cam_K = cam_intrinsics[cam]["K"]
                cam_size = cam_intrinsics[cam]["image_size"]
                cam_D = cam_intrinsics[cam]["D"]
                cam_file = cam_files[cam]
                bbox_2d_out_dir = bbox_2d_out_dirs[cam]

                # Project Stationary Object to 2D and get Bounding Box
                objects_2d_bbox, occlusion_ratios, objects_2d_centroid = get_2d_bboxes(
                    objects_centroid,
                    objects_3d_bbox,
                    accumulated_objects_points,
                    os1_to_cam_ext,
                    cam_K,
                    cam_size,
                    cam_D,
                    DIST_THRESHOLD_PROJECT,
                )

                # Write 2D Bbox information
                bbox_2d_out_file = os.path.join(
                    bbox_2d_out_dir,
                    f"2d_bbox_{cam}_{sequence}_{frame}.txt",
                )
                bbox_2d_out_file = pathlib.Path(bbox_2d_out_file)
                with open(bbox_2d_out_file, "w") as f:
                    for idx, uid in enumerate(objects_uid):
                        if np.isnan(objects_2d_bbox[idx]).any():
                            continue
                        class_id = BBOX_CLASS_TO_ID[objects_class[idx]]
                        occlusion = round(occlusion_ratios[idx], 2)
                        (x1, y1, x2, y2) = map(int, objects_2d_bbox[idx])
                        f.write(f"{uid} {class_id} {occlusion} {x1} {y1} {x2} {y2}\n")

                # Visualization
                if not os.path.exists(cam_file):
                    continue

                image = cv2.imread(str(cam_file), cv2.IMREAD_COLOR)

                # Convert occlusion ratio to occlusion id
                occlusion_conditions = [
                    (occlusion_ratios < 0.05),
                    (occlusion_ratios < 0.25),
                    (occlusion_ratios < 0.75),
                    (occlusion_ratios < 1.0),
                ]
                occlusions = np.select(occlusion_conditions, [0, 1, 2, 3], default=4)

                for idx, bbox_2d in enumerate(objects_2d_bbox):
                    x1, y1, x2, y2 = bbox_2d
                    if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
                        continue
                    # Draw 2D Bounding Box based on occlusion
                    occlusion_rgba = OCCLUSION_ID_TO_COLOR[occlusions[idx]]
                    occlusion_bgr = [int(v * 255) for v in occlusion_rgba[:3][::-1]]
                    x1, y1, x2, y2 = map(int, bbox_2d)
                    cv2.rectangle(image, (x1, y1), (x2, y2), occlusion_bgr, 2)
                    # Draw UID on 2D Bounding Box
                    cv2.putText(
                        image,
                        objects_uid[idx],
                        (x1, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 255),
                        2,
                    )
                    # Project object's pointcloud to 2D
                    object_points = accumulated_objects_points[idx]
                    projected_points, _ = project_points_3d_to_2d(
                        object_points, os1_to_cam_ext, cam_K, cam_size, cam_D
                    )
                    for point in projected_points:
                        cv2.circle(image, (int(point[0]), int(point[1])), 1, (0, 255, 0), 1)
                    # Draw centroid
                    cX, cY = objects_2d_centroid[idx]
                    if not np.isnan(cX) and not np.isnan(cY):
                        cv2.circle(image, (int(cX), int(cY)), 5, (0, 0, 255), 5)

                cam_msg = CvBridge().cv2_to_imgmsg(image)
                cam_msg.header.stamp = ts
                cam_pub.publish(cam_msg)

            # 3D Visualization
            # Publish the 3D Bounding Box
            clear_marker_array(bbox_3d_pub)
            bbox_3d_markers = MarkerArray()
            for bbox_3d in bbox_3d_json["3dbbox"]:
                occlusion_rgba = OCCLUSION_ID_TO_COLOR[
                    OCCLUSION_TO_ID[bbox_3d["labelAttributes"]["isOccluded"]]
                ]
                bbox_marker = create_3d_bbox_marker(
                    (
                        bbox_3d["cX"],
                        bbox_3d["cY"],
                        bbox_3d["cZ"],
                        bbox_3d["l"],
                        bbox_3d["w"],
                        bbox_3d["h"],
                        bbox_3d["r"],
                        bbox_3d["p"],
                        bbox_3d["y"],
                    ),
                    "os1",
                    ts,
                    bbox_3d["instanceId"],
                    int(bbox_3d["instanceId"].split(":")[-1]),
                    occlusion_rgba,
                    0.1,
                )
                bbox_3d_markers.markers.append(bbox_marker)
            bbox_3d_pub.publish(bbox_3d_markers)
            # Publish Object UID
            clear_marker_array(object_id_pub)
            text_markers = MarkerArray()
            text_id = 0
            for classname in instance_uid_coords.keys():
                for uid, coords in instance_uid_coords[classname].items():
                    text_marker = create_text_marker(
                        uid,
                        coords,
                        global_frame,
                        ts,
                        uid,
                        text_id,
                        (1.0, 0, 1.0, 1.0),
                        0.5,
                    )
                    text_markers.markers.append(text_marker)
                    text_id += 1
            object_id_pub.publish(text_markers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CODa path and sequence.")
    parser.add_argument(
        "-d", "--dataset_path", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument(
        "-s", "--sequence", type=str, required=True, help="Sequence number"
    )
    parser.add_argument(
        "-v_pc",
        "--visualize_pointcloud",
        action="store_true",
        help="Visualize the pointcloud",
    )
    parser.add_argument(
        "-v_img", "--visualize_image", action="store_true", help="Visualize the image"
    )

    args = parser.parse_args()
    main(args)
