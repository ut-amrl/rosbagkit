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


def show_depth(depth, colormap=cv2.COLORMAP_VIRIDIS):
    assert depth.ndim == 2, f"{depth.shape} != (H, W)"

    if depth.dtype not in [np.float32, np.float64]:
        depth = depth.astype(np.float32)

    normalized_image = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imshow("Depth Image", cv2.applyColorMap(normalized_image, colormap))
    cv2.waitKey(0)
