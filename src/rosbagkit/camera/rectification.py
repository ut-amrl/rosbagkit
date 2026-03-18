import bisect
from pathlib import Path

import cv2
import numpy as np

from rosbagkit.camera.utils import load_camera_params, load_extrinsics


def sync_indices_closest(
    left_timestamps: list[float] | np.ndarray,
    right_timestamps: list[float] | np.ndarray,
    threshold: float = 0.005,
) -> tuple[list[int], list[int], list[float]]:
    left_ts = np.asarray(left_timestamps, dtype=float)
    right_ts = np.asarray(right_timestamps, dtype=float)

    left_indices: list[int] = []
    right_indices: list[int] = []
    synced_timestamps: list[float] = []
    used_right: set[int] = set()

    for left_idx, left_ts_val in enumerate(left_ts):
        pos = bisect.bisect_left(right_ts, left_ts_val)
        best_idx = None
        best_diff = threshold

        for right_idx in (pos - 1, pos):
            if 0 <= right_idx < len(right_ts) and right_idx not in used_right:
                diff = abs(left_ts_val - right_ts[right_idx])
                if diff < best_diff:
                    best_diff = diff
                    best_idx = right_idx

        if best_idx is None:
            continue

        left_indices.append(left_idx)
        right_indices.append(best_idx)
        synced_timestamps.append(float(left_ts_val))
        used_right.add(best_idx)

    return left_indices, right_indices, synced_timestamps


class StereoRectifier:
    def __init__(
        self,
        k_left: np.ndarray,
        d_left: np.ndarray,
        k_right: np.ndarray,
        d_right: np.ndarray,
        image_size: tuple[int, int],
        rotation: np.ndarray,
        translation: np.ndarray,
    ):
        self.image_size = tuple(int(v) for v in image_size)
        self.image_shape = (self.image_size[1], self.image_size[0])
        self.r1, self.r2, self.p1, self.p2, self.q, self.valid_roi1, self.valid_roi2 = cv2.stereoRectify(
            k_left,
            d_left,
            k_right,
            d_right,
            self.image_size,
            rotation,
            translation,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
        )

        self.map_left_x, self.map_left_y = cv2.initUndistortRectifyMap(
            k_left, d_left, self.r1, self.p1, self.image_size, cv2.CV_32FC1
        )
        self.map_right_x, self.map_right_y = cv2.initUndistortRectifyMap(
            k_right, d_right, self.r2, self.p2, self.image_size, cv2.CV_32FC1
        )

    def rectify(self, image: np.ndarray, left: bool = True) -> np.ndarray:
        if image.shape[:2] != self.image_shape:
            raise ValueError(
                "Stereo image shape does not match rectifier calibration size: "
                f"expected={self.image_shape} got={image.shape[:2]}"
            )
        map_x = self.map_left_x if left else self.map_right_x
        map_y = self.map_left_y if left else self.map_right_y
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)


def build_stereo_rectifier(left_calib: str | Path, right_calib: str | Path, extrinsics: str | Path) -> StereoRectifier:
    left_cam = load_camera_params(left_calib)
    right_cam = load_camera_params(right_calib)
    left_size = tuple(int(v) for v in left_cam["img_size"][::-1])
    right_size = tuple(int(v) for v in right_cam["img_size"][::-1])
    if left_size != right_size:
        raise ValueError(f"Stereo calibration files must use the same image size: left={left_size} right={right_size}")

    transform = load_extrinsics(extrinsics)

    return StereoRectifier(
        left_cam["K"], left_cam["D"], right_cam["K"], right_cam["D"], left_size, transform[:3, :3], transform[:3, 3]
    )
