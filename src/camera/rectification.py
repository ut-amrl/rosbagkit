from loguru import logger

import numpy as np
import cv2


class StereoRectifier:
    def __init__(self, K_left, D_left, K_right, D_right, image_size, R, t):
        # Compute the rectification transformations and projection matrices
        self.R1, self.R2, self.P1, self.P2, self.Q, self.valid_roi1, self.valid_roi2 = (
            cv2.stereoRectify(
                K_left,
                D_left,
                K_right,
                D_right,
                image_size,
                R,
                t,
                flags=cv2.CALIB_ZERO_DISPARITY,  # Ensures principal points are aligned
                alpha=0,
            )
        )
        np.set_printoptions(precision=6, suppress=True)

        # Compute the valid ROI (for alpha=0)
        x1, y1, w1, h1 = self.valid_roi1
        x2, y2, w2, h2 = self.valid_roi2
        self.x1 = max(x1, x2)
        self.y1 = max(y1, y2)
        self.x2 = min(x1 + w1, x2 + w2)
        self.y2 = min(y1 + h1, y2 + h2)

        logger.info(
            f"\nR1:\n{self.R1}\nR2:\n{self.R2}"
            f"\nP1:\n{self.P1}\nP2:\n{self.P2}"
            f"\nQ:\n{self.Q}"
        )

        # Precompute rectification maps
        self.map_left_x, self.map_left_y = cv2.initUndistortRectifyMap(
            K_left, D_left, self.R1, self.P1, image_size, cv2.CV_32FC1
        )
        self.map_right_x, self.map_right_y = cv2.initUndistortRectifyMap(
            K_right, D_right, self.R2, self.P2, image_size, cv2.CV_32FC1
        )

    def rectify(self, image, left: bool = True):
        map_x = self.map_left_x if left else self.map_right_x
        map_y = self.map_left_y if left else self.map_right_y
        rectified = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        return rectified
