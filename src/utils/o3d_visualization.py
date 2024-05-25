import threading
import time
import open3d as o3d

from collections.abc import Iterable
import numpy as np
from scipy.spatial.transform import Rotation as R


class O3dVisualizer:
    def __init__(self, window_name="Open3D Visualizer", background_color=(0, 0, 0)):
        self.window_name = window_name
        self.background_color = np.asarray(background_color)
        self.vis = o3d.visualization.Visualizer()
        self.running = False
        self.thread = threading.Thread(target=self.run)

    def add_geometry(self, geometry):
        self.vis.add_geometry(geometry)

    def clear(self):
        self.vis.clear_geometries()

    def start(self):
        self.running = True
        self.thread.start()

    def run(self):
        self.vis.create_window(window_name=self.window_name)
        self.vis.get_render_option().background_color = self.background_color

        while self.vis.poll_events():
            self.vis.update_renderer()
            time.sleep(0.1)

    def close(self):
        self.running = False
        self.thread.join()
        self.vis.destroy_window()


def visualize_pointcloud(pointcloud, window_name="Point Cloud"):
    if isinstance(pointcloud, Iterable):
        pointcloud = np.vstack(pointcloud)

    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(pointcloud)

    o3d.visualization.draw_geometries(
        [pc_o3d], window_name=window_name, width=800, height=600
    )


def create_o3d_3d_bbox(bbox, color, degrees=False):
    """
    Create a 3D bounding box in Open3D

    Args:
        bbox: (9,) 3D bounding box (cX, cY, cZ, l, w, h, roll, pitch, yaw)
        color: RGB color of the bounding box
    """
    assert len(bbox) == 9, f"{bbox.shape} != (9,)"
    center = bbox[:3]
    extent = bbox[3:6]
    rpy = bbox[6:9]

    rot = R.from_euler("xyz", rpy, degrees=degrees).as_matrix()
    obb = o3d.geometry.OrientedBoundingBox(center=center, R=rot, extent=extent)
    obb.color = color
    return obb


def create_o3d_ellipsoid(ellipsoid, color, degree=False):
    """
    Create an ellipsoid in Open3D

    Args:
        ellipsoid: (9,) ellipsoid parameters (cX, cY, cZ, a, b, c, roll, pitch, yaw)
        color: RGB color of the ellipsoid
    """
    assert len(ellipsoid) == 9, f"{ellipsoid.shape} != (9,)"
    center = ellipsoid[:3]
    radii = ellipsoid[3:6]
    rpy = ellipsoid[6:9]

    # Create an ellipsoid
    ell = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)

    # Scale the ellipsoid
    scaling = np.diag([radii[0], radii[1], radii[2], 1.0])
    ell.transform(scaling)

    # Transform the ellipsoid
    Hwe = np.eye(4)
    Hwe[:3, 3] = center
    Hwe[:3, :3] = R.from_euler("xyz", rpy, degrees=degree).as_matrix()
    ell.transform(Hwe)

    ell.compute_vertex_normals()
    ell.paint_uniform_color(color)
    return ell


if __name__ == "__main__":
    vis = O3dVisualizer()
    vis.start()

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=10, origin=[0, 0, 0]
    )
    vis.add_geometry(coord_frame)

    for _ in range(10):
        ellipsoid = np.random.rand(9) * 10
        # ellipsoid = np.array([0, 0, 0, 1.0, 2.0, 3.0, 0, 0, 0.5], dtype=np.float64)
        ell = create_o3d_ellipsoid(ellipsoid, (255, 0, 0))
        vis.add_geometry(ell)
        time.sleep(1)
