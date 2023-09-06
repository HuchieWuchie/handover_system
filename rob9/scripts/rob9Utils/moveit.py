#!/usr/bin/env python3
import rospy
import geometry_msgs.msg
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
import std_msgs.msg
from std_msgs.msg import String, Bool

import moveit_msgs
import geometry_msgs
from moveit_msgs.srv import GetPositionIK, GetPositionIKResponse
from moveit_msgs.msg import RobotTrajectory
from std_msgs.msg import Bool, Float64MultiArray

from rob9.srv import moveitMoveToNamedSrv, moveitMoveToNamedSrvResponse
from rob9.srv import moveitPlanToNamedSrv, moveitPlanToNamedSrvResponse
from rob9.srv import moveitExecuteSrv, moveitExecuteSrvResponse
from rob9.srv import moveitRobotStateSrv, moveitRobotStateSrvResponse
from rob9.srv import moveitPlanToPoseSrv, moveitPlanToPoseSrvResponse
from rob9.srv import moveitPlanFromPoseToPoseSrv, moveitPlanFromPoseToPoseSrvResponse
from rob9.srv import moveitGetJointPositionAtNamed, moveitGetJointPositionAtNamedResponse



def getRobotStateAtPose(pose_msg):
    """ Input:
        pose_msg            - geometry_msgs/Pose

        Output:
        valid               - Bool, is the state valid or not
        state               - RobotState
    """

    initial_state = getCurrentState(0) # moveit requires a start state given

    start_pose_msg = geometry_msgs.msg.PoseStamped()
    start_pose_msg.header.frame_id = "world"
    start_pose_msg.header.stamp = rospy.Time.now()
    start_pose_msg.pose = pose_msg

    ik_request_msg = moveit_msgs.msg.PositionIKRequest()
    ik_request_msg.group_name = "manipulator"
    ik_request_msg.robot_state = initial_state
    ik_request_msg.avoid_collisions = True #False
    ik_request_msg.pose_stamped = start_pose_msg
    ik_request_msg.timeout = rospy.Duration(1.0) #
    ik_request_msg.attempts = 10

    rospy.wait_for_service('/iiwa/compute_ik')
    ik_calculator = rospy.ServiceProxy("/iiwa/compute_ik", GetPositionIK)

    state = ik_calculator(ik_request_msg)
    valid = False
    if state.error_code.val == 1:
        valid = True

    return valid, state.solution

def getInverseKinematicsSolution(initial_state, pose_msg):

    goal_pose_msg = geometry_msgs.msg.PoseStamped()
    goal_pose_msg.header.frame_id = "world"
    goal_pose_msg.header.stamp = rospy.Time.now()
    goal_pose_msg.pose = pose_msg

    ik_request_msg = moveit_msgs.msg.PositionIKRequest()
    ik_request_msg.group_name = "manipulator"
    ik_request_msg.robot_state = initial_state
    ik_request_msg.avoid_collisions = False #False
    ik_request_msg.pose_stamped = goal_pose_msg
    ik_request_msg.timeout = rospy.Duration(0.25) #
    ik_request_msg.attempts = 5

    rospy.wait_for_service('/iiwa/compute_ik')
    ik_calculator = rospy.ServiceProxy("/iiwa/compute_ik", GetPositionIK)

    state = ik_calculator(ik_request_msg)
    valid = False
    if state.error_code.val == 1:
        valid = True

    return valid, state

def moveToNamed(name):

    if not len(name):
        print("ERROR: Specify a named pose")
        return 0

    rospy.wait_for_service("/rob9/moveit/move_to_named")
    tf2Service = rospy.ServiceProxy("/rob9/moveit/move_to_named", moveitMoveToNamedSrv)

    msg = moveitMoveToNamedSrv()
    msg.data = name

    success = tf2Service(msg)

    return success

def execute(plan):

    rospy.wait_for_service("/rob9/moveit/execute")
    tf2Service = rospy.ServiceProxy("/rob9/moveit/execute", moveitExecuteSrv)

    msg = moveitExecuteSrv()
    msg = plan

    success = tf2Service(msg)
    print(success)

    return success

def planToNamed(name):

    rospy.wait_for_service("/rob9/moveit/plan_to_named")
    service = rospy.ServiceProxy("/rob9/moveit/plan_to_named", moveitPlanToNamedSrv)

    msg = moveitPlanToNamedSrv()
    msg.data = name

    response = service(msg)
    return response.plan

def planFromPoseToPose(start_pose, goal_pose):

    rospy.wait_for_service("/rob9/moveit/plan_from_pose_to_pose")
    service = rospy.ServiceProxy("/rob9/moveit/plan_from_pose_to_pose", moveitPlanFromPoseToPoseSrv)

    response = service(start_pose, goal_pose)
    return response.success, response.plan

def planToPose(pose):

    rospy.wait_for_service("/rob9/moveit/plan_to_pose")
    service = rospy.ServiceProxy("/rob9/moveit/plan_to_pose", moveitPlanToPoseSrv)

    response = service(pose)
    return response.success, response.plan

def getCurrentState():

    rospy.wait_for_service("/rob9/moveit/getRobotState")
    tf2Service = rospy.ServiceProxy("/rob9/moveit/getRobotState", moveitRobotStateSrv)

    msg = moveitRobotStateSrv()
    msg.data = True

    state = tf2Service(msg).state

    return state

def getJointPositionAtNamed(target):
    rospy.wait_for_service("/rob9/moveit/getJointPositionAtNamed")
    service = rospy.ServiceProxy("/rob9/moveit/getJointPositionAtNamed", moveitGetJointPositionAtNamed)
    msg = moveitGetJointPositionAtNamed()
    msg.data = target
    response = service(msg)
    return response
