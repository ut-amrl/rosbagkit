import rospy 
import numpy as np

from helpers.ros_viz_utils import *



if __name__ == '__main__':
    rospy.init_node('test_ros_viz_utils')


    frame_id = 'map'
    bbox_3d = np.array([0, 0, 0, 5, 7, 9, 1, 1, 1])
    bbox_3d_marker = create_bbox_3d_marker(bbox_3d, frame_id)
    bbox_3d_marker_pub = rospy.Publisher('/bbox_3d_marker', Marker, queue_size=1, latch=True)
    bbox_3d_marker_pub.publish(bbox_3d_marker)


    ellipsoid = np.array([0, 0, 0, 5, 7, 9, 1, 1, 1])
    ellipsoid_marker = create_ellipsoid_marker(ellipsoid, frame_id)
    ellipsoid_marker_pub = rospy.Publisher('/ellipsoid_marker', Marker, queue_size=1, latch=True)
    ellipsoid_marker_pub.publish(ellipsoid_marker)

    rospy.spin()
