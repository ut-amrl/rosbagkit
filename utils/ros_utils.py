"""
Author:      Dongmyeong Lee (domlee[at]utexas.edu)
Date:        Nov 21, 2023
Description: ROS utility functions
"""

import os
import signal
import time
import subprocess
import socket
import rospy
from typing import Union, Optional


def start_roscore(new_instance: bool = False):
    # Check if a roscore is already running on the default port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("localhost", 11311)) == 0:
            if not new_instance:
                print("roscore is already running.")
                return None
            print("Attempting to kill the existing roscore instances...")
            kill_roscore()

    print("Starting a new roscore instance...")
    roscore_process = subprocess.Popen(["roscore"])
    time.sleep(5)  # Add a sleep of 3 seconds
    return roscore_process


def kill_roscore():
    for process in ["roscore", "rosmaster"]:
        try:
            pids = (
                subprocess.check_output(["pgrep", "-f", process])
                .decode()
                .strip()
                .split()
            )
            for pid in pids:
                os.kill(int(pid), signal.SIGTERM)
                # os.waitpid(int(pid), 0)
            print("Successfully killed all roscore processes.")
        except subprocess.CalledProcessError:
            print("No roscore processes found to kill.")


def play_bagfile(
    bagfile: str, use_sim_time: bool = True, silent: bool = False
) -> subprocess.Popen:

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
