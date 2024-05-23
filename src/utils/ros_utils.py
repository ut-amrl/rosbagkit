"""
Author:      Dongmyeong Lee (domlee[at]utexas.edu)
Date:        Nov 21, 2023
Description: ROS utility functions
"""

import time
import subprocess
from typing import Union, Optional

import rospy
import rosbag
import tf2_ros


def play_bagfile(
    bagfile: str, use_sim_time: bool = True, silent: bool = False
) -> subprocess.Popen:
    print(f"Playing bagfile: {bagfile}")

    if use_sim_time:
        rospy.set_param("use_sim_time", True)

    command = ["rosbag", "play", "--clock", bagfile]
    if silent:
        command.append("-q")

    return subprocess.Popen(command)


def wait_for_subscribers(
    publishers: Union[rospy.Publisher, list[rospy.Publisher]],
    timeout: Optional[rospy.Duration] = None,
) -> bool:
    if not isinstance(publishers, list):
        publishers = [publishers]

    use_sim_time = rospy.get_param("/use_sim_time", False)

    rate = rospy.Rate(1) if not use_sim_time else None
    wait_duration = 0.1

    for publisher in publishers:
        start_time = rospy.Time.now()
        is_message_logged = False

        while publisher.get_num_connections() == 0:
            if not is_message_logged:
                rospy.loginfo(f"Waiting for subscriber to connect to {publisher.name}")
                is_message_logged = True

            # Sleep
            if not use_sim_time:
                rate.sleep()
            else:
                time.sleep(wait_duration)

            # Timeout
            if timeout and rospy.Time.now() - start_time > timeout:
                rospy.logerr(f"Timeout {timeout.to_sec()}s reached")
                return False
        rospy.loginfo("Subscriber connected to {}".format(publisher.name))

    rospy.loginfo("All subscribers connected to all publishers")

    return True


def log_tf(tf_buffer: tf2_ros.Buffer, bagfile: str, time_limit: Optional[float] = None):
    bag = rosbag.Bag(bagfile)
    start_time = None
    timestamps = []

    for _, msg, t in bag.read_messages(topics=["/tf", "/tf_static"]):
        timestamps.append(t.to_sec())
        if start_time is None:
            start_time = t
        if (t - start_time).to_sec() > time_limit:
            break

        for transform in msg.transforms:
            tf_buffer.set_transform(transform, "default_authority")

    bag.close()
    return timestamps
