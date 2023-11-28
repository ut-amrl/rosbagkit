"""
Author:      Dongmyeong Lee (domlee[at]utexas[dot]edu)
Date:        November 13, 2023
Description: Get instance ID of the object from 2D bounding box
             Backward projection center of the 2D bounding box to 3D
"""
import os
import pathlib
import argparse
from tqdm import tqdm

import numpy as np

import rospy
import cv2
from cv_bridge import CvBridge


import tf2_ros
from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path

from helpers.msg_converter import (
    np_to_pointcloud2,
    pose_stamped_from_quat,
    tf_msg_from_quat,
)
from helpers.coda_utils import load_extrinsic_matrix, load_camera_params

# Frames
global_frame = "map"
lidar_frame = "base_link"
cam_list = ["cam0", "cam1"]


def get_parser():
    parser = argparse.ArgumentParser(description="Get instance ID for CODa")
    parser.add_argument(
        "--dataset",
        type=str,
        default="/home/dongmyeong/Projects/AMRL/CODa",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--pointcloud_topic",
        type=str,
        default="/points_raw",
        help="Pointcloud topic name",
    )
    parser.add_argument(
        "--image_topic",
        type=str,
        default="/image_raw",
        help="Image topic name",
    )
    return parser


def process_frame(
    args,
    pc_pub,
    img_pubs,
    obj_pubs,
    pc_file,
    img_files,
    bbox_2d_files,
    ts,
    os1_to_cam_extrinsics,
    cam_intrinsics,
):
    # Publish Pointcloud
    if os.path.exists(pc_file):
        pc_np = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 4)
        pc_msg = np_to_pointcloud2(pc_np, "x y z i", lidar_frame, ts)
        pc_pub.publish(pc_msg)

    # Publish Images
    for cam in cam_list:
        img_file = img_files[cam]
        bbox_2d_file = bbox_2d_files[cam]

        if os.path.exists(img_file) and os.path.exists(bbox_2d_file):
            continue

        # Draw Bounding Box
        img = cv2.imread(str(img_file))
        bbox_2d = np.loadtxt(bbox_2d_file, delimiter=" ").reshape(-1, 6)
        for bbox in bbox_2d:
            class_id, confidence, x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(
                img,
                f"{int(class_id)} {confidence:.2f}",
                (int(x1), int(y2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

        img_msg = CvBridge().cv2_to_imgmsg(img, "bgr8")
        img_pubs[cam].publish(img_msg)

    # TODO namespace for insatnce id



def main(args):
    rospy.init_node("CODa_instance_getter")

    # Raw Data Publishers
    pc_pub = rospy.Publisher(args.pointcloud_topic, PointCloud2, queue_size=1)
    img_pubs = {
        cam: rospy.Publisher(f"/{cam}/image_rect", Image, queue_size=1)
        for cam in cam_list
    }
    # Pose & Path Publisher
    pose_pub = rospy.Publisher("/pose", PoseStamped, queue_size=1)
    path_pub = rospy.Publisher("/global_path", Path, queue_size=1)
    # Global Path Marker
    global_path = Path()
    global_path.header.frame_id = global_frame
    # Define TF Broadcaster
    tf_broadcaster = tf2_ros.TransformBroadcaster()

    # Back-projected Object Detection Publisher
    object_pubs = {
        "tree": rospy.Publisher("/tree", Point, queue_size=1),
        "pole": rospy.Publisher("/pole", Point, queue_size=1),
    }

    # Data Path
    dataset_path = pathlib.Path(args.dataset)

    # Main Loop
    rate = rospy.Rate(10)
    for sequence in range(3):
        pose_file = dataset_path / "poses" / "dense" / f"{sequence}.txt"
        pose_np = np.loadtxt(pose_file, delimiter=" ").reshape(-1, 8)

        # Data Path
        pc_root_dir = dataset_path / "3d_comp" / "os1" / str(sequence)
        img_root_dirs = {
            cam: dataset_path / "2d_rect" / cam / str(sequence) for cam in cam_list
        }
        bbox_2d_root_dirs = {
            cam: dataset_path / "2d_bbox" / cam / str(sequence) for cam in cam_list
        }

        # Calibration
        calibration_dir = dataset_path / "calibrations" / str(sequence)
        os1_to_cam_extrinsics = {
            cam: load_extrinsic_matrix(calibration_dir / f"calib_os1_to_{cam}.yaml")
            for cam in cam_list
        }
        cam_intrinsics = {
            cam: load_camera_params(calibration_dir / f"calib_{cam}_intrinsics.yaml")
            for cam in cam_list
        }

        # Main Loop
        for frame, pose in tqdm(enumerate(pose_np), miniters=1):
            # Get Pose
            ts, x, y, z, qw, qx, qy, qz = pose
            ts = rospy.Time.from_sec(ts)

            # Publish LiDAR Pose and Path
            pose_msg = pose_stamped_from_quat(pose[1:], global_frame, ts)
            global_path.poses.append(pose_msg)
            pose_pub.publish(pose_msg)
            path_pub.publish(global_path)

            # Publish TF
            tf_msg = tf_msg_from_quat(pose[1:], global_frame, lidar_frame, ts)
            tf_broadcaster.sendTransform(tf_msg)

            # Data Path
            pc_file = pc_root_dir / f"3d_comp_os1_{sequence}_{frame}.bin"
            img_files = {
                cam: img_root_dirs[cam] / f"2d_rect_{cam}_{sequence}_{frame}.jpg"
                for cam in cam_list
            }
            bbox_2d_files = {
                cam: bbox_2d_root_dirs[cam] / f"2d_bbox_{cam}_{sequence}_{frame}.txt"
                for cam in cam_list
            }

            # Process Frame
            process_frame(
                args,
                pc_pub,
                img_pubs,
                obj_pubs,
                pc_file,
                img_files,
                bbox_2d_files,
                ts,
                os1_to_cam_extrinsics,
                cam_intrinsics,
            )
            rate.sleep()


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
