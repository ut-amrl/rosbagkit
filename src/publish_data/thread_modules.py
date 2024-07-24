import time
import threading
import numpy as np
import rospy
import cv2

from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2
import tf2_ros
from cv_bridge import CvBridge

from src.utils.msg_converter import (
    np_to_pointcloud2,
    np_to_imu,
    odometry_from_xyz_quat,
    pose_stamped_from_xyz_quat,
    tf_msg_from_quat,
)


class SharedClock:
    def __init__(self):
        self.lock = threading.Lock()
        self.current_time = None

    def set_time(self, time):
        with self.lock:
            self.current_time = time

    def get_time(self):
        with self.lock:
            return self.current_time


def publish_clock(clock_pub, shared_clock, timestamps, rate=1):
    interval = rospy.Duration.from_sec(1.0 / 1000)  # 1kHz
    first_time = rospy.Time.from_sec(timestamps[0] - 3.0)
    curr_time = first_time
    last_time = rospy.Time.from_sec(timestamps[-1] + 3.0)
    total_duration = (last_time - curr_time).to_sec()

    while curr_time < last_time and not rospy.is_shutdown():
        clock_pub.publish(Clock(curr_time))
        shared_clock.set_time(curr_time)

        elapsed_time = (curr_time - first_time).to_sec()
        progress = elapsed_time / total_duration * 100
        print(f"Progress: {progress:.2f}%", end="\r")
        time.sleep(interval.to_sec())
        curr_time += interval * rate


def publish_imu(imu_pub, imu_data, shared_clock):
    imu_last_idx = 0
    while imu_last_idx < len(imu_data) and not rospy.is_shutdown():
        current_time = shared_clock.get_time()
        if current_time:
            while (
                imu_last_idx < len(imu_data)
                and imu_data[imu_last_idx][0] < current_time.to_sec()
            ):
                ts = rospy.Time.from_sec(imu_data[imu_last_idx][0])
                imu_msg = np_to_imu(imu_data[imu_last_idx][1:], "imu_link", ts)
                imu_pub.publish(imu_msg)
                imu_last_idx += 1
        time.sleep(0.001)


def publish_pointcloud(
    pc_pub, pc_files, timestamps, frame_id, shared_clock, compensated=False, pc_size=3, blind=0.0,
):
    assert len(pc_files) == len(timestamps)

    pc_last_idx = 0
    while pc_last_idx < len(pc_files) and not rospy.is_shutdown():
        current_time = shared_clock.get_time()
        if current_time:
            while (
                pc_last_idx < len(timestamps)
                and timestamps[pc_last_idx] < current_time.to_sec()
            ):
                pc_file = pc_files[pc_last_idx]
                dt = (
                    timestamps[pc_last_idx + 1] - timestamps[pc_last_idx]
                    if pc_last_idx + 1 < len(timestamps)
                    else timestamps[pc_last_idx] - timestamps[pc_last_idx - 1]
                )
                ts = rospy.Time.from_sec(timestamps[pc_last_idx])
                if compensated:
                    pc_np = np.fromfile(pc_file, dtype=np.float32).reshape(-1, pc_size)

                    # filter out points with range less than blind
                    valid = np.linalg.norm(pc_np[:, :2], axis=1) > blind
                    pc_np = pc_np[valid]

                    # add intensity placeholder
                    if pc_size == 3:
                        pc_np = np.hstack(
                            (pc_np, np.zeros((len(pc_np), 1), dtype=np.float32))
                        )
                    pc_msg = np_to_pointcloud2(pc_np, "x y z intensity", frame_id, ts)
                else:
                    pc_msg = process_raw_pointcloud(pc_file, dt * 1e9, frame_id, ts)
                pc_pub.publish(pc_msg)
                pc_last_idx += 1
        time.sleep(0.001)


def process_raw_pointcloud(bin_path, dt, frame_id, timestamp):
    """simulate raw pointcloud data from a binary file"""
    N_HORIZON = 1024
    N_RING = 128

    try:
        points = np.fromfile(bin_path, dtype=np.float32).reshape(N_RING, N_HORIZON, -1)
    except IOError:
        raise IOError("Could not read the file:", bin_path)

    # ring number
    ring_values = np.arange(N_RING)
    ring_values = np.tile(ring_values, (N_HORIZON, 1)).T

    # time between horizontal scans
    t_values = np.linspace(0, dt, N_HORIZON, endpoint=False)
    t_values = np.tile(t_values, (N_RING, 1))

    # placehold for reflectivity, ambient, range
    reflectivity = np.zeros((N_RING, N_HORIZON, 1), dtype=np.float32)
    ambient = np.zeros((N_RING, N_HORIZON, 1), dtype=np.float32)
    # range is distance from the sensor to the point (mm)
    rng = np.zeros((N_RING, N_HORIZON, 1), dtype=np.float32)

    # Add ring and time information to points
    points = np.dstack((points, ring_values, t_values, reflectivity, ambient, rng))

    return np_to_pointcloud2(
        points, "x y z intensity ring t reflectivity ambient range", frame_id, timestamp
    )


def publish_odom(odom_pub, path_pub, pose_np, frame_id, child_frame_id, shared_clock):
    global_path = Path()
    global_path.header.frame_id = frame_id

    pose_last_idx = 0
    while pose_last_idx < len(pose_np) and not rospy.is_shutdown():
        current_time = shared_clock.get_time()
        if current_time:
            while (
                pose_last_idx < len(pose_np)
                and pose_np[pose_last_idx][0] < current_time.to_sec()
            ):
                ts = rospy.Time.from_sec(pose_np[pose_last_idx][0])
                odom_msg = odometry_from_xyz_quat(
                    pose_np[pose_last_idx][1:4],
                    pose_np[pose_last_idx][4:8],
                    frame_id,
                    child_frame_id,
                    ts,
                )
                odom_pub.publish(odom_msg)

                pose_msg = pose_stamped_from_xyz_quat(
                    pose_np[pose_last_idx][1:4],
                    pose_np[pose_last_idx][4:8],
                    frame_id,
                    ts,
                )
                global_path.poses.append(pose_msg)
                path_pub.publish(global_path)
                pose_last_idx += 1
        time.sleep(0.001)


def publish_tf(pose_np, frame_id, child_frame_id, shared_clock):
    tf_broadcaster = tf2_ros.TransformBroadcaster()
    pose_last_idx = 0
    while pose_last_idx < len(pose_np) and not rospy.is_shutdown():
        current_time = shared_clock.get_time()
        if current_time:
            while (
                pose_last_idx < len(pose_np)
                and pose_np[pose_last_idx][0] < current_time.to_sec()
            ):
                ts = rospy.Time.from_sec(pose_np[pose_last_idx][0])
                tf_msg = tf_msg_from_quat(
                    pose_np[pose_last_idx][1:4],
                    pose_np[pose_last_idx][4:8],
                    frame_id,
                    child_frame_id,
                    ts,
                )
                tf_broadcaster.sendTransform(tf_msg)
                pose_last_idx += 1
        time.sleep(0.001)


def publish_static_map(map_file, frame_id):
    static_map_pub = rospy.Publisher(
        "/static_map", PointCloud2, queue_size=1, latch=True
    )
    static_map = np.fromfile(map_file, dtype=np.float32).reshape(-1, 3)
    static_map_msg = np_to_pointcloud2(static_map, "x y z", frame_id)
    static_map_pub.publish(static_map_msg)


def publish_image(image_pub, image_files, timestamps, shared_clock):
    assert len(image_files) == len(timestamps)
    bridge = CvBridge()

    img_last_idx = 0
    while img_last_idx < len(image_files) and not rospy.is_shutdown():
        current_time = shared_clock.get_time()
        if current_time:
            while (
                img_last_idx < len(image_files)
                and timestamps[img_last_idx] < current_time.to_sec()
            ):
                ts = rospy.Time.from_sec(timestamps[img_last_idx])
                img = cv2.imread(str(image_files[img_last_idx]))
                img_msg = bridge.cv2_to_imgmsg(img, encoding="bgr8")
                img_msg.header.stamp = ts
                image_pub.publish(img_msg)
                img_last_idx += 1
        time.sleep(0.001)
