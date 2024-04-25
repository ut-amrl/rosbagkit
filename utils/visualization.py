import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
from collections.abc import Iterable


def visualize_pointcloud(pointcloud, window_name="Point Cloud"):
    if isinstance(pointcloud, Iterable):
        pointcloud = np.vstack(pointcloud)

    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(pointcloud)

    o3d.visualization.draw_geometries(
        [pc_o3d], window_name=window_name, width=800, height=600
    )


def draw_points_on_image(image, points, depths=None):
    if image is None:
        raise ValueError("Image is None")

    if points is None or len(points) == 0:
        print("No points to draw")

    if depths is not None and len(depths) != len(points):
        raise ValueError("Number of points and depths do not match")

    image = image.copy()

    # Check if depth data is available
    if depths is not None:
        # Normalize the logarithmic depth values to the range 0-1
        log_depths = np.log1p(depths - np.min(depths) + 1)
        normalized_depths = (log_depths - np.min(log_depths)) / (
            np.max(log_depths) - np.min(log_depths)
        )
        # perceptually uniform colormap
        colormap = plt.get_cmap("viridis")
        colors = (colormap(normalized_depths) * 255).astype(int)
        for (x, y), color in zip(points.astype(int), colors):
            cv2.circle(image, (x, y), 3, color[:3][::-1].tolist(), -1)
    else:
        for x, y in points.astype(int):
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def normalize_colormap(image, colormap=cv2.COLORMAP_VIRIDIS):
    if image.dtype not in [np.float32, np.float64]:
        image = image.astype(np.float32)

    norm_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return cv2.applyColorMap(norm_image, colormap)


def visualize_normalized_image(image, colormap=cv2.COLORMAP_VIRIDIS, outfile=None):
    norm_image = normalize_colormap(image, colormap)

    if outfile:
        cv2.imwrite(outfile, norm_image)
    else:
        cv2.imshow("Normalized Image", norm_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def visualize_rgbd_image(
    rgb, depth, alpha=0.5, colormap=cv2.COLORMAP_VIRIDIS, outfile=None
):
    norm_depth = normalize_colormap(depth, colormap)
    rgbd_image = cv2.addWeighted(rgb, alpha, norm_depth, 1 - alpha, 0)

    if outfile:
        cv2.imwrite(outfile, rgbd_image)
    else:
        cv2.imshow("RGBD Image", rgbd_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
