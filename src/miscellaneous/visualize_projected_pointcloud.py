"""
Author: Dongmyeong Lee (domlee[at]utexas.edu)
Date:   May 23, 2024
Description: visualize the projected pointcloud on to the image (for debugging)
"""

import argparse
import pathlib
from natsort import natsorted
import threading
import tkinter as tk
from tkinter import font, ttk

import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.utils.camera import load_extrinsics, load_cam_params
from src.utils.projection import project_to_rectified, project_to_image
from src.utils.pose_interpolator import PoseInterpolator
from src.utils.extrinsic_calibrator import ExtrinsicCalibrator
from src.utils.lie_math import xyz_quat_to_matrix


class OffsetTuner:
    def __init__(self):
        self.offset = 0.0
        self.lock = threading.Lock()

        self.gui_thread = threading.Thread(target=self.create_gui)
        self.gui_thread.daemon = True
        self.gui_thread.start()

    def create_gui(self):
        self.root = tk.Tk()
        self.root.title("Offset Tuner")

        self.font = font.Font(size=20)

        self.create_entry("Offset", self.offset)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.root.mainloop()

    def create_entry(self, label, initial_value):
        frame = ttk.Frame(self.root)
        frame.pack(fill="x", padx=5, pady=5)

        label_widget = ttk.Label(frame, text=label, font=self.font)
        label_widget.pack(side="left")

        entry = ttk.Entry(frame, font=self.font)
        entry.insert(0, str(initial_value))
        entry.pack(side="left")

        entry.bind("<KeyRelease>", self.on_parameter_change)
        setattr(self, f"{label.lower()}_entry", entry)

    def on_parameter_change(self, event):
        try:
            new_value = float(self.offset_entry.get())
            self.lock.acquire()
            self.offset = new_value
            self.lock.release()
        except ValueError:
            pass  # Ignore invalid input

    def on_closing(self):
        self.root.quit()
        self.root.destroy()

    def get_offset(self):
        self.lock.acquire()
        offset_value = self.offset
        self.lock.release()
        return offset_value


def alternating_indices(window_size):
    """Generate alternating indices from -window_size//2 to window_size//2

    Example: window_size=7 -> [0, -1, 1, -2, 2, -3, 3]
    """
    yield 0
    for i in range(1, window_size // 2 + 1):
        yield -i
        yield i


def main(args):
    print(f"Visualizing projected pointcloud for {args.scene}")

    # Load the point cloud files and poses
    pc_files = natsorted(args.pc_dir.glob("*.bin"))
    poses = np.loadtxt(args.pose_file)  # timestamp, x, y, z, qw, qx, qy, qz
    assert len(pc_files) == len(poses), f"{len(pc_files)} != {len(poses)}"
    print(f"Loaded {len(pc_files)} pointclouds and poses")

    # Load the images and timestamps
    img_files = natsorted(args.img_dir.glob("*.jpg"))
    img_times = np.loadtxt(args.img_times)
    assert len(img_files) == len(img_times), f"{len(img_files)} != {len(img_times)}"
    print(f"Loaded {len(img_files)} images and timestamps")

    # Load the calibrations
    extrinsic = load_extrinsics(args.cam_extrinsics)
    cam_params = load_cam_params(args.cam_intrinsics)

    # Interpolate the LiDAR poses to the image timestamps
    pose_interpolator = PoseInterpolator(poses)

    idx = 1000
    offset_tuner = OffsetTuner()

    for img_file, img_ts in zip(img_files[idx:], img_times[idx:]):
        # Get the corresponding pointcloud
        close_pc_idx = np.argmin(np.abs(poses[:, 0] - img_ts))
        pc_ts = poses[close_pc_idx, 0]

        # Accumulate the pointclouds
        accumulated_pc = np.zeros((0, 3))
        for i in alternating_indices(args.window_size):
            pc_idx = close_pc_idx + i
            if pc_idx < 0 or pc_idx >= len(pc_files):
                continue

            pc = np.fromfile(str(pc_files[pc_idx]), dtype=np.float32).reshape(-1, 3)
            # pc = pc[:, :3]
            hwl = xyz_quat_to_matrix(poses[pc_idx, 1:])

            # pc_world = pc @ hwl[:3, :3].T + hwl[:3, 3].T
            pc_world = pc
            accumulated_pc = np.vstack((accumulated_pc, pc_world))

        print(f"Accumulated {len(accumulated_pc)} points")
        print(f"image: {img_file.stem}, pc: {pc_files[close_pc_idx].stem}")

        # Project the accumulated pointcloud
        while True:
            offset = offset_tuner.get_offset()

            # get interpolated the pose to the image timestamp
            Hwt = pose_interpolator.get_interpolated_transform(img_ts + offset)
            Htw = np.linalg.inv(Hwt)
            # pc_curr = accumulated_pc @ Htw[:3, :3].T + Htw[:3, 3].T
            pc_curr = accumulated_pc

            if args.img_type == "raw":
                pc_img, depth, _ = project_to_image(
                    pc_curr,
                    extrinsic,
                    cam_params["K"],
                    cam_params["D"],
                    cam_params["img_size"],
                )
            elif args.img_type == "undistorted":
                pc_img, depth, _ = project_to_image(
                    pc_curr,
                    extrinsic,
                    cam_params["K"],
                    np.zeros(5),
                    cam_params["img_size"],
                )
            elif args.img_type == "rectified":
                pc_img, depth, _ = project_to_rectified(
                    pc_curr,
                    extrinsic,
                    cam_params["P"],
                    cam_params["img_size"],
                )
            else:
                raise ValueError("Invalid img_type")

            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
            depth_colormap = plt.get_cmap("jet")(depth_normalized)

            # Draw the projected pointcloud
            image = cv2.imread(str(img_file))
            for (x, y), color in zip(pc_img, depth_colormap):
                x, y = int(x), int(y)
                r, g, b, _ = (np.array(color) * 255).astype(np.uint8)
                cv2.circle(image, (x, y), 1, (int(r), int(g), int(b)), -1)

            cv2.imshow("Projected Pointcloud", image)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


def get_args():
    parser = argparse.ArgumentParser(description="Visualize projected pointcloud")
    parser.add_argument(
        "--dataset", type=str, choices=["CODa", "wanda", "wilbur", "trevor"]
    )
    parser.add_argument("--dataset_dir", type=str, help="Path to the dataset")
    parser.add_argument("--scene", type=str, help="Scene name")
    parser.add_argument(
        "--img_type", type=str, choices=["raw", "rectified", "undistorted"]
    )
    parser.add_argument("--window_size", type=int, default=1, help="Window size (odd)")
    args = parser.parse_args()

    args.dataset_dir = pathlib.Path(args.dataset_dir)
    if args.dataset == "CODa":
        args.pc_dir = args.dataset_dir / "3d_comp" / "os1" / args.scene
        args.pose_file = args.dataset_dir / "correct" / f"{args.scene}.txt"
        args.img_dir = args.dataset_dir / "2d_rect" / "cam0" / args.scene
        args.img_times = args.dataset_dir / "timestamps" / f"{args.scene}.txt"
        args.calib_dir = args.dataset_dir / "calibrations" / "0"
        args.cam_intrinsics = args.calib_dir / "calib_cam0_intrinsics.yaml"
        args.cam_extrinsics = args.calib_dir / "calib_os1_to_cam0.yaml"
    elif args.dataset == "wanda":
        args.pc_dir = args.dataset_dir / "3d_comp" / args.scene
        args.pose_file = args.dataset_dir / "poses" / args.scene / "os1.txt"
        args.img_dir = args.dataset_dir / "2d_rect" / args.scene / "left"
        args.img_times = args.dataset_dir / "timestamps" / args.scene / "img_left.txt"
        args.calib_dir = args.dataset_dir / "calibrations" / args.scene
        args.cam_intrinsics = args.calib_dir / "cam_left_intrinsics.yaml"
        args.cam_extrinsics = args.calib_dir / "os_to_cam_left.yaml"
    elif args.dataset == "wilbur":
        args.pc_dir = args.dataset_dir / "3d_comp" / args.scene
        args.pose_file = args.dataset_dir / "poses" / args.scene / "os1.txt"
        args.img_dir = args.dataset_dir / "2d_rect" / args.scene
        args.img_times = args.dataset_dir / "timestamps" / args.scene / "img_aux.txt"
        args.calib_dir = args.dataset_dir / "calibrations" / args.scene
        args.cam_intrinsics = args.calib_dir / "cam_aux_intrinsics.yaml"
        args.cam_extrinsics = args.calib_dir / "os_to_cam_aux.yaml"
    elif args.dataset == "warthog":
        args.pc_dir = args.dataset_dir / "3d_comp" / args.scene
        args.pose_file = args.dataset_dir / "poses" / args.scene / "os1.txt"
        args.img_dir = args.dataset_dir / "2d_rect" / args.scene / "front"
        args.img_times = (
            args.dataset_dir / "timestamps" / args.scene / "img_aux_front.txt"
        )
        args.calib_dir = args.dataset_dir / "calibrations" / args.scene
        args.cam_intrinsics = args.calib_dir / "cam_aux_front_intrinsics.yaml"
        args.cam_extrinsics = args.calib_dir / "os_to_cam_aux_front.yaml"
    else:
        raise ValueError("Invalid dataset")

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
