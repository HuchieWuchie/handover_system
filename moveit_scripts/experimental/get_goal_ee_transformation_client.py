#!/usr/bin/env python
import rospy
import geometry_msgs.msg
from moveit_scripts.srv import *
import numpy as np
from tf.transformations import *

def get_random_transformation():
    transform = geometry_msgs.msg.Transform()
    transform.translation.x = np.random.uniform(low=0.5, high=1.5)
    transform.translation.y = np.random.uniform(low=0.5, high=1.5)
    transform.translation.z = np.random.uniform(low=0.5, high=1.5)

    #roll = np.random.uniform(low=0.0, high=3.1416)
    #pitch = np.random.uniform(low=0.0, high=3.1416)
    #yaw = np.random.uniform(low=0.0, high=3.1416)
    #q = quaternion_from_euler(roll, pitch, yaw)

    q = np.zeros(4)
    q[0] = np.random.uniform(low=0.0, high=1.0) #q_x
    q[1] = np.random.uniform(low=0.0, high=1.0) #q_y
    q[2] = np.random.uniform(low=0.0, high=1.0) #q_z
    q[3] = np.random.uniform(low=0.0, high=1.0) #q_w

    # normalize quaternion MAY BE UNNECESSARY IF YOU USE quaternion_from_euler()
    q_mag = np.linalg.norm(q)
    q = q/q_mag

    # turn numpy array q to geometry_msgs
    transform.rotation.x = q[0]
    transform.rotation.y = q[1]
    transform.rotation.z = q[2]
    transform.rotation.w = q[3]
    return transform


if __name__ == '__main__':
    rospy.init_node('get_goal_ee_transformation_client_node', anonymous=True)
    rospy.wait_for_service('get_goal_ee_transformation')
    get_goal_ee  = rospy.ServiceProxy("get_goal_ee_transformation", GetGoalEeTransformation)

    req = GetGoalEeTransformationRequest()
    res = GetGoalEeTransformationResponse()

    req.grasp = get_random_transformation()
    req.centroid = get_random_transformation()
    req.goal_centroid = get_random_transformation()
    print(req)
    res = get_goal_ee(req)

    if(res.success.data == True):
        print("=========================")
        print(res)
