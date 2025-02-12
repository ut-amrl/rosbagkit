from loguru import logger

import numpy as np
import cv2


def write_depth(depth, filename):
    """
    Save the depth image with uint16 format. (unit: millimeters)

    Args:
        depth: The depth image to save (unit: meters)
        filename: The filename to save the depth image
    """
    assert depth.ndim == 2, f"{depth.shape} != (H, W)"

    depth = depth.copy()

    # Convert the depth image to millimeters
    depth[np.isnan(depth) | np.isinf(depth) | (depth < 0) | (depth > 65.535)] = 0
    depth = np.clip(depth * 1000, 0, 65535).astype(np.uint16)

    # Save the depth image
    cv2.imwrite(filename, depth)


def load_depth(filename):
    """
    load the depth image with uint16 format. (unit: millimeters)

    Args:
        filename: The filename to read the depth image
    """
    depth = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    depth = depth.astype(np.float32) / 1000
    return depth


def visualize_depth(depth, colormap=cv2.COLORMAP_TURBO, path=""):
    assert depth.ndim == 2, f"{depth.shape} != (H, W)"

    if depth.dtype not in [np.float32, np.float64]:
        depth = depth.astype(np.float32)

    nonzero_mask = depth > 0

    # Initialize the color_depth image with zeros (black)
    color_depth = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

    if np.any(nonzero_mask):
        # Replace zero depth values with a small epsilon to prevent division by zero
        # Not strictly necessary here since we're masking, but good practice
        epsilon = 1e-6
        depth_safe = np.where(nonzero_mask, depth, epsilon)

        # Compute inverse depth
        inverse_depth = 1.0 / depth_safe

        # Define visualization range for inverse depth
        # Adjust these values based on your specific depth range and requirements
        max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)  # min: 0.1m
        min_invdepth_vizu = max(inverse_depth.min(), 1 / 65)  # max: 65m
        inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
            max_invdepth_vizu - min_invdepth_vizu
        )


        # Apply the colormap to the normalized inverse depth
        colored_inv_depth = cv2.applyColorMap(
            (inverse_depth_normalized * 255).astype(np.uint8), colormap
        )

        # Assign the colored inverse depth to the color_depth image using the mask
        color_depth[nonzero_mask] = colored_inv_depth[nonzero_mask]

    if path:
        cv2.imwrite(path, color_depth)
    else:
        cv2.imshow("Depth Image", color_depth)
        cv2.waitKey(0)
