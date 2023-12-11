"""
Author:      Dongmyeong Lee (domlee[at]utexas.edu)
Date:        Dec 11, 2023
Description: Accumualate converted laserscan data from map (.pcd) file
"""
import argparse
from pathlib import Path
from natsort import natsorted
from tqdm import tqdm

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
import open3d

import rospy
from sensor_msgs.msg import PointCloud2
from utils.msg_converter import np_to_pointcloud2


def get_parser():
    parser = argparse.ArgumentParser(description="Convert map file to 2D for ENML")
    parser.add_argument(
        "--map",
        type=str,
        default="/home/dongmyeong/Projects/AMRL/CODa/correction/ut_campus_downsampled.pcd",
        help="Path to map file (.pcd)",
    )
    parser.add_argument(
        "--poses",
        type=str,
        default="/home/dongmyeong/Projects/AMRL/CODa/poses/keyframe",
        help="Path to poses file (.txt)",
    )
    parser.add_argument(
        "--min_height",
        type=float,
        default=1.5,
        help="Minimum height of point cloud to be considered (LiDAR frame)",
    ),
    parser.add_argument(
        "--max_height",
        type=float,
        default=2.0,
        help="Maximum height of point cloud to be considered (LiDAR frame)",
    )
    parser.add_argument(
        "--min_dist",
        type=float,
        default=0.5,
        help="Minimum distance of point cloud to be considered (LiDAR frame)",
    )
    parser.add_argument(
        "--max_dist",
        type=float,
        default=30.0,
        help="Maximum distance of point cloud to be considered (LiDAR frame)",
    )
    parser.add_argument(
        "--num_ranges",
        type=int,
        default=360,
        help="Number of ranges in the laserscan",
    )
    return parser


def main(args):
    # Load Map
    pcd_map = open3d.io.read_point_cloud(args.map)
    pcd_map_np = np.asarray(pcd_map.points)
    # KDTree for fast radius search
    pcd_map_tree = KDTree(pcd_map_np)

    rospy.init_node("map_2d", anonymous=True)
    pub = rospy.Publisher("/map_2d", PointCloud2, queue_size=10)

    # 2D pointcloud for ENML localization
    pointcloud_2d = np.zeros((0, 3))
    # Main loop
    pose_files = natsorted(Path(args.poses).glob("*.txt"))
    for pose_file in pose_files:
        pose_np = np.loadtxt(pose_file, delimiter=" ")[:, :8]

        for pose in tqdm(pose_np, total=len(pose_np)):
            pose_mat = np.eye(4)
            pose_mat[:3, :3] = R.from_quat(pose[[5, 6, 7, 4]]).as_matrix()
            pose_mat[:3, 3] = pose[1:4]

            # Query point cloud
            query_idx = pcd_map_tree.query_ball_point(pose_mat[:3, 3], r=args.max_dist)
            queried_pcd_map = pcd_map_np[query_idx]

            # Transform point cloud to LiDAR frame
            queried_pcd_map_lidar = (
                np.column_stack((queried_pcd_map, np.ones(queried_pcd_map.shape[0])))
                @ np.linalg.inv(pose_mat).T
            )

            # Convert to laserscan
            ranges = np.inf * np.ones(args.num_ranges)
            for point in queried_pcd_map_lidar:
                if point[2] < args.min_height or point[2] > args.max_height:
                    continue

                dist = np.linalg.norm(point[:2])
                if dist < args.min_dist or dist > args.max_dist:
                    continue

                angle = np.arctan2(point[1], point[0])
                idx = int(np.round((angle + np.pi) / (2 * np.pi) * args.num_ranges))
                idx = idx % args.num_ranges
                ranges[idx] = min(ranges[idx], dist)

            # Convert laserscan to pointcloud in map frame
            pc_2d = np.zeros((args.num_ranges, 4))  # homogeneous coord in LiDAR frame
            pc_2d[:, 0] = ranges * np.cos(np.linspace(-np.pi, np.pi, args.num_ranges))
            pc_2d[:, 1] = ranges * np.sin(np.linspace(-np.pi, np.pi, args.num_ranges))
            pc_2d[:, 3] = 1.0
            pc_2d = pc_2d[ranges != np.inf] @ pose_mat.T  # Transform to map frame

            # Save point cloud
            pointcloud_2d = np.vstack((pointcloud_2d, pc_2d[:, :3]))
            pc2_msg = np_to_pointcloud2(pointcloud_2d[:, :3], "x y z", "map")
            pub.publish(pc2_msg)

        break

    # Save point cloud
    pcd_2d = open3d.geometry.PointCloud()
    pcd_2d.points = open3d.utility.Vector3dVector(pointcloud_2d)
    out_file = Path(args.map).parent / f"{Path(args.map).stem}_2d.pcd"
    open3d.io.write_point_cloud(out_file, pcd_2d)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
