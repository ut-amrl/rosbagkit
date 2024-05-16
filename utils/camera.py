import sys
import pathlib
import numpy as np


def project_to_image(img, pc, cam_ext, K, D):
    """
    Project the point cloud to the image plane

    Args:
        img: The image to project the point cloud onto.
        pc: (N,3) The point cloud to project.
        cam_ext: (4,4) The camera extrinsic matrix.
        K: (3,3) The camera intrinsic matrix.
        D: (5,) The distortion coefficients.

    Returns:
        pc_img: (M,2) The projected point cloud in the image plane.
        pc_depth: (M,) The depth values of the projected points.
        valid_indices: (M,) The indices of the valid points.
    """
    H, W = img.shape[:2]

    # Transform points to camera coordinate system
    pc_cam = np.hstack((pc, np.ones((len(pc), 1))))
    pc_cam = pc_cam @ cam_ext[:3].T
    # 0 < z < 10
    valid_pc = np.logical_and(pc_cam[:, 2] > 0, pc_cam[:, 2] < 10)

    # Project points onto the image plane
    pc_img = pc_cam[valid_pc, :2] / pc_cam[valid_pc, 2, None]

    # Apply radial distortion
    x, y = pc_img[:, 0], pc_img[:, 1]
    r2 = x**2 + y**2
    radial = 1 + D[0] * r2 + D[1] * r2**2 + D[4] * r2**3
    tan_x = 2 * D[2] * x * y + D[3] * (r2 + 2 * x**2)
    tan_y = D[2] * (r2 + 2 * y**2) + 2 * D[3] * x * y
    x = x * radial + tan_x
    y = y * radial + tan_y
    pc_img = np.vstack((x, y, np.ones_like(x))).T

    # Apply camera matrix
    pc_img = pc_img @ K[:2].T

    # Filter points outside the image
    in_bound = np.logical_and(
        np.logical_and(pc_img[:, 0] >= 0, pc_img[:, 0] < W),
        np.logical_and(pc_img[:, 1] >= 0, pc_img[:, 1] < H),
    )

    valid_indices = np.where(valid_pc)[0][in_bound]

    return pc_img[in_bound, :2], pc_cam[valid_indices, 2], valid_indices


def project_to_rectified(img, pc, cam_ext, R, P):
    """
    Project the point cloud to the rectified image

    Args:
        img: The image to project the point cloud onto.
        pc: (N,3) The point cloud to project.
        cam_ext: (4,4) The camera extrinsic matrix.
        R: (3,3) The rectification matrix.
        P: (3,4) The projection matrix.

    Returns:
        pc_img: (M,2) The projected point cloud in the image plane.
        pc_depth: (M,) The depth values of the projected points.
        valid_indices: (M,) The indices of the valid points.
    """
    H, W = img.shape[:2]

    # Transform points to camera coordinate system
    pc_cam = np.hstack((pc, np.ones((len(pc), 1))))  # (N, 4) homogeneous coordinates
    pc_cam = pc_cam @ cam_ext[:3].T  # (N, 3) camera coordinates
    valid_pc = np.logical_and(pc_cam[:, 2] > 0, pc_cam[:, 2] < 10)

    # Project points onto the image plane
    pc_img = pc_cam[valid_pc] @ R.T
    pc_img = np.hstack((pc_img, np.ones((len(pc_img), 1))))
    pc_img = pc_img @ P.T
    pc_img = pc_img[:, :2] / pc_img[:, 2, None]

    # Filter points outside the image
    in_bound = np.logical_and(
        np.logical_and(pc_img[:, 0] >= 0, pc_img[:, 0] < W),
        np.logical_and(pc_img[:, 1] >= 0, pc_img[:, 1] < H),
    )

    valid_indices = np.where(valid_pc)[0][in_bound]

    return pc_img[in_bound], pc_cam[valid_indices, 2], valid_indices


def get_visible_cloud(pointcloud: np.ndarray, voxel_size=(0.1, 0.1, 0.1)):
    import pcl

    pc_pcl = pcl.PointCloud()
    pc_pcl.from_array(pointcloud)
    pc_pcl.sensor_origin = np.array([0, 0, 0, 1], dtype=np.float32)
    pc_pcl.sensor_orientation = np.array([0, 0, 0, 1], dtype=np.float32)
    voxel_filter = pcl.VoxelGridOcclusionEstimation.PointXYZ()
    voxel_filter.setInputCloud(pc_pcl)
    voxel_filter.setLeafSize(*voxel_size)
    voxel_filter.initializeVoxelGrid()
    occupied_voxels = pcl.vectors.PointXYZ()
    voxel_filter.getOccupiedVoxels(occupied_voxels)

    visible_cloud = np.array(occupied_voxels)


if __name__ == "__main__":
    # random point cloud
    pointcloud = np.random.rand(100, 3).astype(np.float32)
    get_visible_cloud(pointcloud)
