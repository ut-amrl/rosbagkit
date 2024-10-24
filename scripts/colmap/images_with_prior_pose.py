"""
Description: Generate "images.txt" for COLMAP to reconstruct a sparse model from known camera poses
"""

import os
import argparse
from collections import defaultdict

import sqlite3
import numpy as np

from pose.utils import xyz_quat_to_matrix, matrix_to_xyz_quat

def get_args():
    parser = argparse.ArgumentParser(description="Generate images.txt for COLMAP")
    parser.add_argument("--database_path", default="database.db")
    parser.add_argument("--cam_pose_files", nargs="+", default=["cam0.txt", "cam1.txt"])
    parser.add_argument("--output_file", default="images.txt")
    args = parser.parse_args()

    if not os.path.exists(args.database_path):
        raise FileNotFoundError(f"{args.database_path} does not exist.")

    return args


def main(args):
    # Connect to the database
    db = sqlite3.connect(args.database_path)
    cursor = db.cursor()

    # Query to get image_id, camera_id, and name from the images table
    cursor.execute("SELECT image_id, camera_id, name FROM images")
    images_db = cursor.fetchall()

    camera_ids_to_image = defaultdict(list)
    for image_id, camera_id, name in images_db:
        camera_ids_to_image[int(camera_id)].append((image_id, camera_id, name))
    camera_ids = sorted(camera_ids_to_image.keys())

    # sanity check
    assert len(camera_ids) == len(args.cam_pose_files)

    # Read camera poses
    camera_poses = {}
    for idx, cam_pose_file in enumerate(args.cam_pose_files):
        camera_id = camera_ids[idx]
        poses = np.loadtxt(cam_pose_file)  # [ts, x, y, z, qw, qx, qy, qz]
        assert len(poses) == len(camera_ids_to_image[camera_id])
        camera_poses[camera_id] = iter(poses)

    with open(args.output_file, "w") as f:
        for image_id, camera_id, name in images_db:
            pose = next(camera_poses[int(camera_id)])
            extrinsic = np.linalg.inv(xyz_quat_to_matrix(pose[1:]))
            tx, ty, tz, qw, qx, qy, qz = matrix_to_xyz_quat(extrinsic)

            # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            line = f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {name}\n"
            f.write(line)
            f.write("\n")

    # Close the database
    db.close()


if __name__ == "__main__":
    args = get_args()
    main(args)
