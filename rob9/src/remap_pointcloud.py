#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2

def pc_callback(msg):
    msg.header.frame_id = "ptu_camera_color_optical_frame_real"
    pc_pub.publish(msg)


if __name__ == '__main__':
    rospy.init_node('remap_pointcloud', anonymous=True)
    rospy.Subscriber('sensors/realsense/pointcloudGeometry/static', PointCloud2, pc_callback)
    pc_pub = rospy.Publisher('sensors/realsense/pointcloudGeometry/static_corrected', PointCloud2, queue_size=1)

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
