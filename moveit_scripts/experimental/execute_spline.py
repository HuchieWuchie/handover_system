#!/usr/bin/env python3
import sys
import copy
import rospy
import math
import numpy as np
import time
import open3d as o3d
import cv2
import random
import sys, signal
from scipy.spatial.transform import Rotation as R
#import actionlib

import geometry_msgs.msg
from geometry_msgs.msg import Pose, PoseStamped, PoseArray, Transform
import std_msgs.msg
from std_msgs.msg import Int8, MultiArrayDimension, MultiArrayLayout, Int32MultiArray, Float32MultiArray, Bool, Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

import rob9Utils.transformations as transform
from rob9Utils.graspGroup import GraspGroup
from rob9Utils.grasp import Grasp
import rob9Utils.moveit as moveit
from cameraService.cameraClient import CameraClient
from affordanceService.client import AffordanceClient
from grasp_service.client import GraspingGeneratorClient
from orientationService.client import OrientationClient
from locationService.client import LocationClient
from rob9Utils.visualize import visualizeGrasps6DOF, visualizeMasksInRGB
import rob9Utils.iiwa
from rob9Utils.visualize import visualizeMasksInRGB, visualizeFrameMesh, createGripper, visualizeGripper
from rob9Utils.affordancetools import getPredictedAffordances, getAffordanceContours, getObjectAffordancePointCloud, getPointCloudAffordanceMask
from rob9Utils.utils import erodeMask, keepLargestContour, convexHullFromContours, maskFromConvexHull, thresholdMaskBySize, removeOverlapMask


from moveit_scripts.srv import *
from moveit_scripts.msg import *
from rob9.srv import graspGroupSrv, graspGroupSrvResponse

import time

def pertubateEEPose(pose, iteration, rate = 0.005):


    pose.position.x += random.uniform(0, iteration * rate)
    pose.position.y += random.uniform(0, iteration * rate)
    pose.position.z += random.uniform(0, iteration * rate)

    pose.orientation.x += random.uniform(0, iteration * (rate * 0.1))
    pose.orientation.y += random.uniform(0, iteration * (rate * 0.1))
    pose.orientation.z += random.uniform(0, iteration * (rate * 0.1))
    pose.orientation.w += random.uniform(0, iteration * (rate * 0.1))

    return pose

def signal_handler(signal, frame):
    print("Shutting down program.")
    sys.exit()

signal.signal(signal.SIGINT, signal_handler)

def send_trajectory_to_rviz(plan):
    print("Trajectory was sent to RViZ")
    display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    display_trajectory.trajectory_start = moveit.getCurrentState()
    display_trajectory.trajectory.append(plan)
    display_trajectory_publisher.publish(display_trajectory)


def callback(msg):
    global req_obj_id, req_aff_id

    req_obj_id = msg.data[0]
    req_aff_id = msg.data[1]

def computeWaypoint(grasp, offset = 0.1):
    """ input:  graspsObjects   -   rob9Utils.grasp.Grasp() in world_frame
                offset          -   float, in meters for waypoint in relation to grasp
        output:
				waypoint		-   rob9Utils.grasp.Grasp()
    """

    world_frame = "world"
    ee_frame = "right_ee_link"

    waypoint = copy.deepcopy(grasp)

	# you can implement some error handling here if the grasp is given in the wrong frame
	#waypointWorld = Grasp().fromPoseStampedMsg(transform.transformToFrame(waypointCamera.toPoseStampedMsg(), world_frame))
	#graspWorld = Grasp().fromPoseStampedMsg(transform.transformToFrame(graspCamera.toPoseStampedMsg(), world_frame))

    # computing waypoint in camera frame
    rotMat = grasp.getRotationMatrix()
    offsetArr = np.array([[0.0], [0.0], [offset]])
    offsetCam = np.transpose(np.matmul(rotMat, offsetArr))[0]

    waypoint.position.x += -offsetCam[0]
    waypoint.position.y += -offsetCam[1]
    waypoint.position.z += -offsetCam[2]

    return waypoint

def wait():
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    global grasps_affordance, img, affClient, pcd, masks, bboxs, req_aff_id, req_obj_id, state

    print("Init")
    rospy.init_node('moveit_subscriber', anonymous=True)

    state = 1 # start at setup phase
    rate = rospy.Rate(10)

    while True:

        if state == 1:
            # setup phase

            set_ee = True
            if not rob9Utils.iiwa.setEndpointFrame():
                set_ee = False
            print("STATUS end point frame was changed: ", set_ee)

            set_PTP_speed_limit = True
            if not rob9Utils.iiwa.setPTPJointSpeedLimits(0.4, 0.4):
                set_PTP_speed_limit = False
            print("STATUS PTP joint speed limits was changed: ", set_PTP_speed_limit)

            set_PTP_cart_speed_limit = True
            if not rob9Utils.iiwa.setPTPCartesianSpeedLimits(0.4, 0.4, 0.4, 0.4, 0.4, 0.4):
                set_PTP_cart_speed_limit = False
            print("STATUS PTP cartesian speed limits was changed: ", set_PTP_cart_speed_limit)

            #rospy.Subscriber('tool_id', Int8, callback)
            rospy.Subscriber('objects_affordances_id', Int32MultiArray, callback )
            gripper_pub = rospy.Publisher('iiwa/gripper_controller', Int8, queue_size=10, latch=True)
            pub_grasp = rospy.Publisher('iiwa/pose_to_reach', PoseStamped, queue_size=10)
            pub_waypoint = rospy.Publisher('iiwa/pose_to_reach_waypoint', PoseStamped, queue_size=10)
            display_trajectory_publisher = rospy.Publisher('iiwa/move_group/display_planned_path',
                                                moveit_msgs.msg.DisplayTrajectory,
                                                queue_size=20)
            # DO NOT REMOVE THIS SLEEP, it allows gripper_pub to establish connection to the topic
            rospy.sleep(0.1)
            rospy.sleep(2)

            vid_capture = cv2.VideoCapture(0)

            reset_gripper_msg = std_msgs.msg.Int8()
            reset_gripper_msg.data = 0
            activate_gripper_msg = std_msgs.msg.Int8()
            activate_gripper_msg.data = 1
            close_gripper_msg = std_msgs.msg.Int8()
            close_gripper_msg = 2
            open_gripper_msg = std_msgs.msg.Int8()
            open_gripper_msg.data = 3
            basic_gripper_msg = std_msgs.msg.Int8()
            basic_gripper_msg.data = 4
            pinch_gripper_msg = std_msgs.msg.Int8()
            pinch_gripper_msg.data = 5

            gripper_pub.publish(reset_gripper_msg)
            gripper_pub.publish(activate_gripper_msg)
            gripper_pub.publish(open_gripper_msg)
            gripper_pub.publish(pinch_gripper_msg)
            rob9Utils.iiwa.execute_spline_trajectory(moveit.planToNamed("ready"))
            #moveit.execute(moveit.planToNamed("ready"))
            #moveit.moveToNamed("handover")
            #moveit.moveToNamed("ready")
            exit()

            req_obj_id = -1
            req_aff_id = -1

            print("Services init")

            state = 2

        elif state == 2:
			# Capture sensor information

            print("Camera is capturing new scene")

            cam = CameraClient()
            cam.captureNewScene()
            cloud, cloudColor = cam.getPointCloudStatic()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cloud)
            pcd.colors = o3d.utility.Vector3dVector(cloudColor)

            cloud_uv = cam.getUvStatic()
            img = cam.getRGB()

            state = 3

        elif state == 3:
            # Analyze affordance

            print("Segmenting affordance maps")
            affClient = AffordanceClient()

            affClient.start(GPU=True)
            _ = affClient.run(img, CONF_THRESHOLD = 0.7)

            masks, labels, scores, bboxs = affClient.getAffordanceResult()
            masks = affClient.processMasks(masks, conf_threshold = 0, erode_kernel=(1,1))

            cv2.imshow("masks", visualizeMasksInRGB(img, masks))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            state = 4
            print("Found: ", labels, " waiting for command")

        elif state == 4:
            while True:
                rate.sleep()
                if req_aff_id != -1 and req_obj_id != -1:
                    break
            print("Attempting to pick up: ", req_obj_id, " with affordance: ", req_aff_id)
            state = 5

        elif state == 5:
            # Check user input
            num_occurences = np.where(labels == req_obj_id)[0].shape[0]
            if num_occurences == 0:
                req_obj_id = -1
                req_obj_id = -1
                state = 2

            state = 6

        elif state == 6:
            # post process affordance segmentation maps

            obj_inst = np.where(labels == req_obj_id)[0][0]
            obj_inst_masks = masks[obj_inst]
            obj_inst_labels = labels[obj_inst]
            obj_inst_bbox = bboxs[obj_inst]

            # Post process affordance predictions and compute point cloud affordance mask

            affordances_in_object = getPredictedAffordances(masks = obj_inst_masks, bbox = obj_inst_bbox)
            print("predicted affordances", affordances_in_object)

            cv2.imshow("masks", visualizeMasksInRGB(img, obj_inst_masks))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            for aff in affordances_in_object:

                m_vis = np.zeros(obj_inst_masks.shape)

                masks = erodeMask(affordance_id = aff, masks = obj_inst_masks,
                                kernel = np.ones((3,3)))
                contours = getAffordanceContours(bbox = obj_inst_bbox, affordance_id = aff,
                                                masks = obj_inst_masks)


                if len(contours) > 0:
                    contours = keepLargestContour(contours)
                    hulls = convexHullFromContours(contours)

                    h, w = obj_inst_masks.shape[-2], obj_inst_masks.shape[-1]
                    if obj_inst_bbox is not None:
                        h = int(obj_inst_bbox[3] - obj_inst_bbox[1])
                        w = int(obj_inst_bbox[2] - obj_inst_bbox[0])

                    aff_mask = maskFromConvexHull(h, w, hulls = hulls)
                    _, keep = thresholdMaskBySize(aff_mask, threshold = 0.05)
                    if keep == False:
                        aff_mask[:, :] = False

                    if obj_inst_bbox is not None:
                        obj_inst_masks[aff, obj_inst_bbox[1]:obj_inst_bbox[3], obj_inst_bbox[0]:obj_inst_bbox[2]] = aff_mask
                        m_vis[aff, obj_inst_bbox[1]:obj_inst_bbox[3], obj_inst_bbox[0]:obj_inst_bbox[2]] = aff_mask
                    else:
                        obj_inst_masks[aff, :, :] = aff_mask
                        m_vis[aff,:,:] = aff_mask

                    cv2.imshow("masks", visualizeMasksInRGB(img, m_vis))
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()


            obj_inst_masks = removeOverlapMask(masks = obj_inst_masks)

            affordances_in_object = getPredictedAffordances(masks = obj_inst_masks, bbox = obj_inst_bbox)
            print("predicted affordances after post processing", affordances_in_object)

            cv2.imshow("masks", visualizeMasksInRGB(img, obj_inst_masks))
            cv2.waitKey(0)
            cv2.destroyAllWindows()


            state = 7

        elif state == 7:
            # transform point cloud into world coordinate frame
            # below is a transformation used during capture of sample data

            T, translCam2World, rotMatCam2World = transform.getTransform("ptu_camera_color_optical_frame", "world")
            _, t_c2w, r_c2w = transform.getTransform("world", "ptu_camera_color_optical_frame")

            quat_world_to_put = transform.quaternionFromRotation(r_c2w)
            world_to_put_rot = R.from_matrix(rotMatCam2World)
            quat_world_to_put = world_to_put_rot.as_quat()

            tf_msg = Transform()
            tf_msg.translation.x = t_c2w[0]
            tf_msg.translation.y = t_c2w[1]
            tf_msg.translation.z = t_c2w[2]

            tf_msg.rotation.x = quat_world_to_put[0]
            tf_msg.rotation.y = quat_world_to_put[1]
            tf_msg.rotation.z = quat_world_to_put[2]
            tf_msg.rotation.w = quat_world_to_put[3]

            print(tf_msg)

            transform.visualizeTransform(tf_msg, "world_to_camera")




            #cam2WorldTransform = np.array([-(math.pi/2) - 1.0995574287564276, 0, -(math.pi)-(math.pi/4)+1.2220795422464295])
            #rotCam2World = R.from_euler('xyz', cam2WorldTransform)
            #rotMatCam2World = rotCam2World.as_matrix()

            #points = np.dot(rotMatCam2World, np.asanyarray(pcd.points).T).T + translCam2World
            pcd.transform(T)
            points = np.asanyarray(pcd.points)
            #pcd.points = o3d.utility.Vector3dVector(points)
            pcd_affordance = getObjectAffordancePointCloud(pcd, obj_inst_masks, uvs = cloud_uv)

            # Compute a downsampled version of the point cloud for collision checking
            # downsampling speeds up computation
            pcd_downsample = pcd.voxel_down_sample(voxel_size=0.005)

            state = 8

        elif state == 8:

            # Select affordance mask to compute grasps for
            observed_affordances = getPredictedAffordances(obj_inst_masks)

            success, sampled_grasp_points = getPointCloudAffordanceMask(affordance_id = req_aff_id,
                                            points = points, uvs = cloud_uv, masks = obj_inst_masks)

            if success:

                # computing goal pose of object in world frame and
                # current pose of object in world frame

                rotClient = OrientationClient()
                current_orientation, current_position, goal_orientation_giver = rotClient.getOrientation(pcd_affordance) # we discard translation

                curr_rot_quat_world = R.from_matrix(current_orientation).as_quat()
                curr_pose_world = np.hstack((current_position.flatten(), curr_rot_quat_world))
                transform.visualizeTransform(transform.poseToTransform(curr_pose_world), "object_current_pose")
                rotated_coordinate_frame = visualizeFrameMesh(current_position, current_orientation)
                #rotated_coordinate_frame_to_goal = visualizeFrameMesh(translation, goal)

                o3d.visualization.draw_geometries([pcd, rotated_coordinate_frame])

                ## Get goal position using linescan service

                _, _, rotMatGiver2World = transform.getTransform("giver", "world")
                goal_orientation_world = np.matmul(np.linalg.inv(rotMatGiver2World), goal_orientation_giver)


                goal_rot_quat = R.from_matrix(goal_orientation_world).as_quat()

                loc_client = LocationClient()
                goal_location = loc_client.getLocation().flatten()
                goal_location[0] = 0.4

                goal_pose_world = np.hstack((goal_location.flatten(), goal_rot_quat))

                #goal_pose_world = transform.transformToFrame(goal_pose_giver, "world", "giver")
                transform.visualizeTransform(transform.poseToTransform(goal_pose_world), "object_goal_pose")

                # run the grasp algorithm
                grasp_client = GraspingGeneratorClient()
                grasp_group = grasp_client.run(sampled_grasp_points, pcd_downsample,
                                            "world", req_obj_id, req_aff_id, obj_inst)

                print("I got ", len(grasp_group.grasps), " grasps")

                #grasp_group.thresholdByScore(0.3)
                grasp_group.sortByScore()

                for grasp in grasp_group:

                    waypoint = computeWaypoint(grasp, offset = 0.1)
                    waypoint_msg = waypoint.toPoseMsg()

                    plan_found_waypoint, plan_waypoint = moveit.planToPose(waypoint_msg)
                    if len(plan_waypoint.joint_trajectory.points) > 0:
                        grasp_msg = grasp.toPoseMsg()
                        plan_found_waypoint_to_grasp, plan_grasp = moveit.planFromPoseToPose(waypoint_msg, grasp_msg)

                        if len(plan_grasp.joint_trajectory.points) > 0:

                            pub_waypoint.publish(waypoint.toPoseStampedMsg())
                            pub_grasp.publish(grasp.toPoseStampedMsg())



                            # Compute the homegenous 4x4 transformation matrices

                            world_grasp_T = transform.poseStampedToMatrix(grasp.toPoseStampedMsg()) # grasp_pose_world
                            world_centroid_T = transform.poseToMatrix(curr_pose_world)
                            world_centroid_T_goal = transform.poseToMatrix(goal_pose_world)
                            #world_centroid_T_goal =

                            # Compute an end effector pose that properly orients the grasped tool

                            grasp_world_T = np.linalg.inv(world_grasp_T)
                            grasp_centroid_T = np.matmul(grasp_world_T, world_centroid_T)

                            centroid_grasp_T = np.linalg.inv(grasp_centroid_T)
                            world_grasp_T_goal = np.matmul(world_centroid_T_goal, centroid_grasp_T)
                            goal_q = transform.quaternionFromRotation(world_grasp_T_goal)

                            world_centroid_T_test = np.matmul(world_grasp_T, grasp_centroid_T)
                            world_centroid_T_goal_test = np.matmul(world_grasp_T_goal, grasp_centroid_T)

                            # Create poseStamped ros message

                            ee_goal_msg = geometry_msgs.msg.PoseStamped()
                            ee_goal_msg.header.frame_id = "world"
                            ee_goal_msg.header.stamp = rospy.Time.now()

                            ee_pose = Pose()
                            ee_pose.position.x = world_grasp_T_goal[0,3]
                            ee_pose.position.y = world_grasp_T_goal[1,3]
                            ee_pose.position.z = world_grasp_T_goal[2,3]

                            ee_pose.orientation.x = goal_q[0]
                            ee_pose.orientation.y = goal_q[1]
                            ee_pose.orientation.z = goal_q[2]
                            ee_pose.orientation.w = goal_q[3]

                            ee_goal_msg.pose = ee_pose

                            ee_tf = Transform()
                            ee_tf.translation = ee_pose.position
                            ee_tf.rotation = ee_pose.orientation

                            transform.visualizeTransform(ee_tf, "goal_EE_pose")

                            plan_handover = 0
                            # call planToPose with ee_goal_msg
                            i = 0
                            while i < 100:
                                ee_pertubated = pertubateEEPose(ee_pose, i, rate = 0.005)
                                plan_found_handover, plan_handover = moveit.planToPose(ee_pertubated)
                                print(i, ee_pertubated)

                                if len(plan_handover.joint_trajectory.points) > 0:
                                    i = 100
                                i += 1



                            print("Num points in waypoint plan: ", len(plan_waypoint.joint_trajectory.points))
                            print("Num points in grasp plan: ", len(plan_grasp.joint_trajectory.points))
                            print("Num points in handover plan: ", len(plan_handover.joint_trajectory.points))

                            if len(plan_handover.joint_trajectory.points) > 0:
                            #if plan_found_handover:

                                pos = grasp.position.getVector()
                                rot = grasp.orientation.getRotationMatrix()

                                gripper_parts = createGripper(opening = 0.08, translation = pos, rotation = rot)
                                vis_gripper = visualizeGripper(gripper_parts)

                                vis_grasps = visualizeGrasps6DOF(pcd, grasp_group)
                                o3d.visualization.draw_geometries([pcd, *vis_grasps, vis_gripper])

                                print("Moving to waypoint...")

                                rob9Utils.iiwa.execute_spline_trajectory(plan_waypoint)
                                #moveit.execute(plan_waypoint)
                                rospy.sleep(1)

                                print("Moving to grasp pose...")

                                rob9Utils.iiwa.execute_spline_trajectory(plan_grasp)
                                #moveit.execute(plan_grasp)
                                rospy.sleep(1)

                                gripper_pub.publish(close_gripper_msg)
                                rospy.sleep(1)

                                print("I have grasped!")
                                print("Moving to ready...")
                                #input("Press Enter when you are ready to move the robot back to the ready pose") # outcommented by Albert Wed 23 March 09:06

                                rob9Utils.iiwa.execute_spline_trajectory(moveit.planToNamed("ready"))
                                #moveit.moveToNamed("ready")

                                # Execute plan to handover pose
                                rob9Utils.iiwa.execute_spline_trajectory(plan_handover)
                                #moveit.execute(plan_handover)
                                rospy.sleep(2)
                                #input("Press Enter when you are ready to move the robot back to the ready pose")

                                gripper_pub.publish(open_gripper_msg)
                                rospy.sleep(2)
                                rob9Utils.iiwa.execute_spline_trajectory(moveit.planToNamed("ready"))
                                #moveit.moveToNamed("ready")
                                break

            state = 9

        elif state == 9:
            # restart
            req_aff_id = -1
            req_obj_id = -1

            state = 2

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    #rotMat = grasp.orientation.getRotationMatrix()
    #translation = grasp.position.getVector()


    #gripper = createGripper(opening = 0.08, translation = translation, rotation = rotMat)
    #vis_gripper = visualizeGripper(gripper)





    #print("I got ", len(grasp_group.grasps), " grasps after thresholding")

    #vis_grasps = visualizeGrasps6DOF(pcd, grasp_group)
    #o3d.visualization.draw_geometries([pcd, *vis_grasps, vis_gripper])
