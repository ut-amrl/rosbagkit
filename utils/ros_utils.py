"""
Author:      Dongmyeong Lee (domlee[at]utexas.edu)
Date:        Nov 21, 2023
Description: ROS utility functions
"""
import time
from typing import Union

import rospy


def wait_for_subscribers(
    publishers: Union[rospy.Publisher, list[rospy.Publisher]],
    timeout: rospy.Duration = None,
) -> bool:
    if not isinstance(publishers, list):
        publishers = [publishers]

    use_sim_time = rospy.get_param("/use_sim_time", False)

    rate = rospy.Rate(10) if not use_sim_time else None
    wait_duration = 0.1

    for publisher in publishers:
        start_time = rospy.Time.now()
        is_message_logged = False

        while publisher.get_num_connections() == 0:
            if not is_message_logged:
                rospy.loginfo(
                    "Waiting for subscriber to connect to {}".format(publisher.name)
                )
                is_message_logged = True

            # Sleep
            if not use_sim_time:
                rate.sleep()
            else:
                time.sleep(wait_duration)

            # Timeout
            if timeout and rospy.Time.now() - start_time > timeout:
                rospy.logerr(
                    "Timeout while waiting for subscriber to connect to {}".format(
                        publisher.name
                    )
                )
                return False
        rospy.loginfo("Subscriber connected to {}".format(publisher.name))

    rospy.loginfo("All subscribers connected to all publishers")

    return True
