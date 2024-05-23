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


def show_image(image, title="Image"):
    cv2.imshow(title, image)
    while True:
        if (
            cv2.waitKey(1) == ord("q")
            or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1
        ):
            break
    cv2.destroyAllWindows()


def visualize_points_on_image(image, points, depths=None, point_size=3, outfile=None):
    assert image is not None, "Image is None"
    assert points.ndim == 2 and points.shape[1] == 2, f"{points.shape} != (N, 2)"
    assert (depths is None) or (depths.ndim == 1), f"{depths.shape} != (N,)"
    assert point_size > 0, "point_size must be a positive number"

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
            cv2.circle(image, (x, y), point_size, color[:3][::-1].tolist(), -1)
    else:
        for x, y in points.astype(int):
            cv2.circle(image, (x, y), point_size, (0, 255, 0), -1)

    if outfile:
        cv2.imwrite(outfile, image)
    else:
        show_image(image, "Points on Image")


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
