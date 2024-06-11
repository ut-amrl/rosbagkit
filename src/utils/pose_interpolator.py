import numpy as np
from typing import Literal
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from src.utils.lie_math import xyz_quat_to_SE3, SE3_to_xyz_quat, xyz_quat_to_matrix


class PoseInterpolator:
    def __init__(
        self, poses: np.ndarray, method: Literal["linear", "slerp"] = "linear"
    ):
        """poses: (N, 8) [time, x, y, z, qw, qx, qy, qz]"""
        assert poses.shape[1] == 8, f"{poses.shape} != (N, 8)"
        assert method in ["linear", "slerp"], f"{method} not in ['linear', 'slerp']"
        self.poses = poses
        self._sort_poses()
        self.method = method

    def _sort_poses(self):
        """Sort the poses based on the time"""
        self.poses = self.poses[self.poses[:, 0].argsort()]

    def _get_interpolated_SE3(self, time):
        lower_pose, upper_pose = self.get_bound_poses(time)

        if lower_pose[0] == upper_pose[0]:
            return xyz_quat_to_SE3(lower_pose[1:])

        lower_SE3 = xyz_quat_to_SE3(lower_pose[1:])
        upper_SE3 = xyz_quat_to_SE3(upper_pose[1:])

        alpha = (time - lower_pose[0]) / (upper_pose[0] - lower_pose[0])

        if self.method == "linear":
            # T_interpolated = T_lower * exp(alpha * log(T_lower^-1 * T_upper))
            interpolated_SE3 = lower_SE3 + alpha * (upper_SE3 - lower_SE3)
        elif self.method == "slerp":
            key_times = [lower_pose[0], upper_pose[0]]
            key_rots = R.from_quat(
                [
                    lower_pose[[5, 6, 7, 4]],
                    upper_pose[[5, 6, 7, 4]],
                ]
            )
            slerp = Slerp(key_times, key_rots)
            t_interp = (1 - alpha) * lower_pose[1:4] + alpha * upper_pose[1:4]
            q_interp = slerp(time).as_quat()[[3, 0, 1, 2]]

            interpolated_SE3 = xyz_quat_to_SE3(np.concatenate([t_interp, q_interp]))
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return interpolated_SE3

    def _get_relative_SE3(self, source_time, target_time):
        source_SE3 = self._get_interpolated_SE3(source_time)
        target_SE3 = self._get_interpolated_SE3(target_time)

        # T_source_to_target = T_target^-1 * T_source
        relative_SE3 = target_SE3.between(source_SE3)
        return relative_SE3

    def add_poses(self, new_poses):
        """Add a pose to the existing poses"""
        if new_poses.shape == (8,):
            new_poses = new_poses.reshape(1, 8)
        assert new_poses.shape[1] == 8, f"{new_poses.shape} != (8,) or (N, 8)"

        self.poses = np.vstack([self.poses, new_poses])
        self._sort_poses()

    def get_bound_poses(self, time):
        """Get the lower and upper poses for the time"""
        idx = np.searchsorted(self.poses[:, 0], time)

        if idx == 0:
            return self.poses[0], self.poses[0]

        if idx == len(self.poses):
            return self.poses[-1], self.poses[-1]

        return self.poses[idx - 1], self.poses[idx]

    def is_time_in_range(self, time):
        """Check if the time is within the pose range"""
        return self.poses[0, 0] <= time <= self.poses[-1, 0]

    def find_closest_index(self, time):
        idx = np.searchsorted(self.poses[:, 0], time)
        if idx == 0:
            return 0
        if idx == len(self.poses):
            return len(self.poses) - 1

        lower_time = self.poses[idx - 1, 0]
        upper_time = self.poses[idx, 0]
        if time - lower_time < upper_time - time:
            return idx - 1

        return idx

    def get_interpolated_pose(self, time):
        """Get the interpolated pose at the time [time, x, y, z, qw, qx, qy, qz]"""
        interpolated_SE3 = self._get_interpolated_SE3(time)
        interpolated_pose = np.array([time] + list(SE3_to_xyz_quat(interpolated_SE3)))
        return interpolated_pose

    def get_interpolated_transform(self, time):
        """Get the interpolated transform at the time [4x4]"""
        interpolated_SE3 = self._get_interpolated_SE3(time)
        return interpolated_SE3.transform()

    def get_relative_transform(self, source_time, target_time):
        """Get the relative transform from source_time to target_time [4x4]"""
        if source_time == target_time:
            return np.eye(4)

        relative_SE3 = self._get_relative_SE3(source_time, target_time)
        return relative_SE3.transform()


if __name__ == "__main__":
    import open3d as o3d

    poses = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )

    pose_interpolator = PoseInterpolator(poses, "slerp")
    a = pose_interpolator.get_relative_transform(0, 1)
    b = pose_interpolator.get_relative_transform(3, 4)

    point = np.array([0.0, 0.0, 0.0, 1.0])
    print(a.dot(point))

    print(pose_interpolator.is_time_in_range(-1))

    print(pose_interpolator.find_closest_index(0.9))

    geometries = []
    for pose in poses:
        transform = xyz_quat_to_matrix(pose[1:])

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        frame.transform(transform)
        geometries.append(frame)

    for time in np.linspace(0, 2, 100):
        interpolated_pose = pose_interpolator.get_interpolated_pose(time)
        transform = xyz_quat_to_matrix(interpolated_pose[1:])

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        frame.transform(transform)
        geometries.append(frame)

    o3d.visualization.draw_geometries(geometries)
