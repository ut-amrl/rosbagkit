import os
import pathlib
import argparse
from collections import defaultdict

import sqlite3
import numpy as np

from colmap.database import COLMAPDatabase


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", default="project")
    parser.add_argument(
        "--image_dirs",
        nargs="+",
        default=["images0", "images1"],
        help="images directories relative to project_path",
    )
    parser.add_argument(
        "--cam_paths",
        nargs="+",
        help="camera parameters file paths",
    )
    parser.add_argument(
        "--cam_pose_files",
        nargs="+",
        help="camera pose files (world to camera)",
    )
    args = parser.parse_args()
    assert len(args.image_dirs) == len(args.cam_pose_files)

    args.project_dir = pathlib.Path(args.project_dir)
    args.database_path = args.project_dir / "database.db"
    args.image_dirs = [args.project_dir / image_dir for image_dir in args.image_dirs]
    args.sparse_model_dir = args.project_dir / "sparse"

    if args.database_path.exists():
        raise FileExistsError(f"{args.database_path} already exists.")

    args.sparse_model_dir.mkdir(parents=True, exist_ok=True)

    return args


def main(args):
    # Open the database
    db = COLMAPDatabase.connect(args.database_path)
    db.create_tables()

    # Create cameras
    camera_ids = []
    for idx in range(len(args.image_dirs)):
        camera_id = db.add_camera()
        camera_ids.append(camera_id)

    # Create images
    for idx, image_dir in enumerate(args.image_dirs):
        camera_id = camera_ids[idx]
        image_files = sorted(image_dir.glob("*.jpg"))
        for image_file in image_files:
            image_id = db.add_image(image_file, camera_id1)

    # Create Sparse model


if __name__ == "__main__":
    args = get_args()
    main(args)
