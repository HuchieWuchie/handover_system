#!/usr/bin/env python
import sys
import copy
import rospy
import moveit_commander
from moveit_commander.conversions import pose_to_list
from moveit_msgs.srv import *
import moveit_msgs.msg
from moveit_msgs.msg import PositionIKRequest, RobotState, MoveItErrorCodes
import geometry_msgs.msg
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
# from geometry_msgs.msg import Pose
import std_msgs.msg
from std_msgs.msg import Int8
from math import pi
import tf2_ros
import tf2_geometry_msgs
import tf_conversions
import tf2_ros


def callback(msg):
    t_list = []
    poses_len = len(msg.poses)
    print ("Poses length " + str(poses_len))
    for i in range(len(msg.poses)):
        child_frame_id = "grasp_to_reach"+str(i)
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = msg.header.frame_id
        t.child_frame_id = child_frame_id
        t.transform.translation.x = msg.poses[i].position.x
        t.transform.translation.y = msg.poses[i].position.y
        t.transform.translation.z = msg.poses[i].position.z
        t.transform.rotation.x = msg.poses[i].orientation.x
        t.transform.rotation.y = msg.poses[i].orientation.y
        t.transform.rotation.z = msg.poses[i].orientation.z
        t.transform.rotation.w = msg.poses[i].orientation.w
        t_list.append(t)
    t_list_len = len(t_list)
    print ("t_list length " + str(t_list_len))
    #br.sendTransform([t_list[0], t_list[1],  t_list[2],  t_list[3]] )
    br.sendTransform(t_list)


if __name__ == '__main__':
    rospy.init_node('moveit_subscriber', anonymous=True)
    rospy.Subscriber('poses_to_reach', PoseArray, callback)

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    br = tf2_ros.TransformBroadcaster()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
