#!/usr/bin/env python
import rospy
import geometry_msgs.msg
from tf.transformations import *
from moveit_scripts.srv import *
import tf
import tf2_ros
import numpy as np

def visualise(transforms, names):
    broadcaster_0 = tf2_ros.StaticTransformBroadcaster()
    broadcaster_1 = tf2_ros.StaticTransformBroadcaster()
    broadcaster_2 = tf2_ros.StaticTransformBroadcaster()
    broadcaster_3 = tf2_ros.StaticTransformBroadcaster()
    rospy.sleep(1.0)
    static_transformStamped = geometry_msgs.msg.TransformStamped()

    static_transformStamped.header.stamp = rospy.Time.now()
    static_transformStamped.header.frame_id = "world"
    static_transformStamped.child_frame_id = names[0]
    static_transformStamped.transform.translation = transforms[0].translation
    static_transformStamped.transform.rotation = transforms[0].rotation
    broadcaster_0.sendTransform(static_transformStamped)

    static_transformStamped.header.stamp = rospy.Time.now()
    static_transformStamped.header.frame_id = "world"
    static_transformStamped.child_frame_id = names[1]
    static_transformStamped.transform.translation = transforms[1].translation
    static_transformStamped.transform.rotation = transforms[1].rotation
    broadcaster_1.sendTransform(static_transformStamped)

    static_transformStamped.header.stamp = rospy.Time.now()
    static_transformStamped.header.frame_id = "world"
    static_transformStamped.child_frame_id = names[2]
    static_transformStamped.transform.translation = transforms[2].translation
    static_transformStamped.transform.rotation = transforms[2].rotation
    broadcaster_2.sendTransform(static_transformStamped)

    static_transformStamped.header.stamp = rospy.Time.now()
    static_transformStamped.header.frame_id = "world"
    static_transformStamped.child_frame_id = names[3]
    static_transformStamped.transform.translation = transforms[3].translation
    static_transformStamped.transform.rotation = transforms[3].rotation
    broadcaster_3.sendTransform(static_transformStamped)

def msg_to_matrix(pose):
    #http://docs.ros.org/en/melodic/api/tf/html/python/transformations.html
    q = pose.rotation
    #https://answers.ros.org/question/324354/how-to-get-rotation-matrix-from-quaternion-in-python/
    # q.x, q.y, q.z, q.w IS THE CORRECT ORDER
    #transformation = quaternion_matrix([q.w, q.x, q.y, q.z])
    transformation = quaternion_matrix([q.x, q.y, q.z, q.w])
    #q_temp = quaternion_from_matrix(transformation)
    #print(name)
    #print(transformation)
    #print("--------------------")
    #print(transformation_2)
    #print(transformation)

    #rot = R.from_quat(pose.orientation)
    #transformation[0, :] = rot[0, :]
    #transformation[1, :] = rot[1, :]
    #transformation[2, :] = rot[2, :]
    #print(transformation)

    transformation[0,3] = pose.translation.x
    transformation[1,3] = pose.translation.y
    transformation[2,3] = pose.translation.z
    #print(transformation)

    #q = pose.orientation
    #rpy = tf.transformations.euler_from_quaternion(q)
    return transformation

def handle_get_goal_ee(req):
    world_grasp_T =msg_to_matrix(req.grasp)
    world_centroid_T =msg_to_matrix(req.centroid)
    world_centroid_T_goal =msg_to_matrix(req.goal_centroid)

    grasp_world_T = np.linalg.inv(world_grasp_T)
    grasp_centroid_T = np.matmul(grasp_world_T, world_centroid_T)
    #print(ee_centroid_mat)

    centroid_grasp_T = np.linalg.inv(grasp_centroid_T)
    world_grasp_T_goal = np.matmul(world_centroid_T_goal, centroid_grasp_T)
    goal_q = quaternion_from_matrix(world_grasp_T_goal)
    #print(world_grasp_T_goal)


    ##################
    world_centroid_T_test = np.matmul(world_grasp_T, grasp_centroid_T)
    world_centroid_T_goal_test = np.matmul(world_grasp_T_goal, grasp_centroid_T)
    print("world_centroid_T && world_centroid_T_test")
    print(world_centroid_T)
    print(world_centroid_T_test)
    print("==========================================")
    print("world_centroid_T_goal && world_centroid_T_goal_test")
    print(world_centroid_T_goal)
    print(world_centroid_T_goal_test)
    print("==========================================")

    grasp_vec = world_grasp_T[:,3]
    tool_vec = world_centroid_T[:,3]
    grasp_tool_dir_vec = grasp_vec - tool_vec
    grasp_tool_dir_vec_mag = np.linalg.norm(grasp_tool_dir_vec)
    grasp_vec_goal = world_grasp_T_goal[:,3]
    tool_vec_goal = world_centroid_T_goal[:,3]
    grasp_tool_dir_vec_goal = grasp_vec_goal - tool_vec_goal
    grasp_tool_dir_vec_mag_goal = np.linalg.norm(grasp_tool_dir_vec_goal)
    print("grasp_tool magnitude && grasp_tool_goal magnitude")
    print(grasp_tool_dir_vec_mag)
    print(grasp_tool_dir_vec_mag_goal)
    ##################


    res = GetGoalEeTransformationResponse()
    res.success.data = True

    res.goal_ee.translation.x = world_grasp_T_goal[0,3]
    res.goal_ee.translation.y = world_grasp_T_goal[1,3]
    res.goal_ee.translation.z = world_grasp_T_goal[2,3]
    res.goal_ee.rotation.x = goal_q[0]
    res.goal_ee.rotation.y = goal_q[1]
    res.goal_ee.rotation.z = goal_q[2]
    res.goal_ee.rotation.w = goal_q[3]

    transforms = [req.grasp, req.centroid, req.goal_centroid, res.goal_ee]
    names = ["1_grasp", "1_tool_pose", "1_goal_tool_pose", "1_goal_end_effector_pose"]
    visualise(transforms, names)

    #print(res)
    return res

if __name__ == '__main__':
    rospy.init_node('get_goal_ee_transformation_server_node', anonymous=True)
    server = rospy.Service('get_goal_ee_transformation', GetGoalEeTransformation, handle_get_goal_ee)
    print("server is ready")

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        print("except")
        pass
