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
import signal
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import os
import argparse

#import actionlib

import geometry_msgs.msg
from geometry_msgs.msg import Pose, PoseStamped, PoseArray, Transform
import std_msgs.msg
from std_msgs.msg import Int8, Int16, MultiArrayDimension, MultiArrayLayout, Int32MultiArray, Float32MultiArray, Bool, Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from iiwa_msgs.msg import JointPosition

import rob9Utils.transformations as transform
from rob9Utils.graspGroup import GraspGroup
from rob9Utils.grasp import Grasp
import rob9Utils.moveit as moveit
from cameraService.cameraClient import CameraClient
from affordanceService.client import AffordanceClient
from grasp_service.client import GraspingGeneratorClient
from orientationService.client import OrientationClient
from locationService.client import LocationClient
from rob9Utils.visualize import visualizeGrasps6DOF, visualizeMasksInRGB, visualizeBBoxInRGB
import rob9Utils.iiwa
from rob9Utils.visualize import visualizeMasksInRGB, visualizeFrameMesh, createGripper, visualizeGripper
from rob9Utils.affordancetools import getPredictedAffordances, getAffordanceContours, getObjectAffordancePointCloud, getPointCloudAffordanceMask
from rob9Utils.utils import erodeMask, keepLargestContour, convexHullFromContours, maskFromConvexHull, thresholdMaskBySize, removeOverlapMask


from moveit_scripts.srv import *
from moveit_scripts.msg import *
from rob9.srv import graspGroupSrv, graspGroupSrvResponse

import time


def signal_handler(signal, frame):
    print("Shutting down program.")
    sys.exit()

signal.signal(signal.SIGINT, signal_handler)


def callback(msg):
    global req_obj_id

    req_obj_id = msg.data[0]

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

def parse_args():

    parser = argparse.ArgumentParser(description='Evaluate the affordance segmentation performance with a weighted F-measure')
    parser.add_argument('--save', dest='save',
                        help='If enabled, saves the various outputs into a folder.',
                        action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    global grasps_affordance, img, pcd, masks, bboxs, req_aff_id, req_obj_id, state

    args = parse_args()

    reset_gripper_msg = std_msgs.msg.Int16()
    reset_gripper_msg.data = 0
    activate_gripper_msg = std_msgs.msg.Int16()
    activate_gripper_msg.data = 1
    close_gripper_msg = std_msgs.msg.Int16()
    close_gripper_msg = 2
    open_gripper_msg = std_msgs.msg.Int16()
    open_gripper_msg.data = 3
    basic_gripper_msg = std_msgs.msg.Int16()
    basic_gripper_msg.data = 4
    pinch_gripper_msg = std_msgs.msg.Int16()
    pinch_gripper_msg.data = 5
    adjust_width_gripper_msg = std_msgs.msg.Int16()
    adjust_width_gripper_msg.data = 120 # 155 is those 8 cm
    increase_force_gripper_msg = std_msgs.msg.Int16()
    increase_force_gripper_msg.data = 30
    increase_speed_gripper_msg = std_msgs.msg.Int16()
    increase_speed_gripper_msg.data = 10


    print("Init")
    rospy.init_node('moveit_subscriber', anonymous=True)

    state = 1 # start at setup phase
    rate = rospy.Rate(10)

    obj_counter = 0
    obj_ids = [11, 10, 16]
    obj_ids = [16]

    while True:

        if state == 1:

            t_string = str(time.time()*1000)
            if not os.path.exists(t_string):
                os.makedirs(t_string)
            # setup phase

            set_ee = True
            if not rob9Utils.iiwa.setEndpointFrame():
                set_ee = False
            print("STATUS end point frame was changed: ", set_ee)

            set_PTP_speed_limit = True
            if not rob9Utils.iiwa.setPTPJointSpeedLimits(0.2, 0.2):
                set_PTP_speed_limit = False
            print("STATUS PTP joint speed limits was changed: ", set_PTP_speed_limit)

            set_PTP_cart_speed_limit = True
            if not rob9Utils.iiwa.setPTPCartesianSpeedLimits(0.2, 0.2, 0.2, 0.2, 0.2, 0.2):
                set_PTP_cart_speed_limit = False
            print("STATUS PTP cartesian speed limits was changed: ", set_PTP_cart_speed_limit)

            #rospy.Subscriber('tool_id', Int8, callback)
            rospy.Subscriber('objects_affordances_id', Int32MultiArray, callback )

            pub_grasp = rospy.Publisher('iiwa/pose_to_reach', PoseStamped, queue_size=10)
            pub_waypoint = rospy.Publisher('iiwa/pose_to_reach_waypoint', PoseStamped, queue_size=10)
            pub_iiwa = rospy.Publisher('iiwa/command/JointPosition', JointPosition, queue_size=10 )
            gripper_pub = rospy.Publisher('iiwa/gripper_controller', Int16, queue_size=10, latch=True)
            display_trajectory_publisher = rospy.Publisher('iiwa/move_group/display_planned_path',
                                                moveit_msgs.msg.DisplayTrajectory,
                                                queue_size=20)
            # DO NOT REMOVE THIS SLEEP, it allows gripper_pub to establish connection to the topic
            #rospy.sleep(0.1)
            rospy.sleep(2)

            vid_capture = cv2.VideoCapture(0)


            gripper_pub.publish(reset_gripper_msg)
            rospy.sleep(0.1)
            gripper_pub.publish(activate_gripper_msg)
            rospy.sleep(0.1)
            gripper_pub.publish(open_gripper_msg)
            rospy.sleep(0.1)
            gripper_pub.publish(pinch_gripper_msg)
            rospy.sleep(0.1)
            gripper_pub.publish(adjust_width_gripper_msg)
            rospy.sleep(0.1)
            for i in range(12):
                gripper_pub.publish(increase_force_gripper_msg)
                rospy.sleep(0.1)
                gripper_pub.publish(increase_speed_gripper_msg)
                rospy.sleep(0.1)


            result = rob9Utils.iiwa.execute_ptp(moveit.getJointPositionAtNamed("ready").joint_position.data)
            state_ready = moveit.getCurrentState()

            print("Services init")

            state = 2

        elif state == 2:

            print("State 2")

            result = rob9Utils.iiwa.execute_ptp(moveit.getJointPositionAtNamed("camera_ready_3").joint_position.data)
            req_obj_id = -1
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

            if args.save:
                o3d.io.write_point_cloud(os.path.join(t_string, str(obj_counter) + "pcd.ply"), pcd)
                np.save(os.path.join(t_string, str(obj_counter) + "uv.npy"), cloud_uv)
                cv2.imwrite(os.path.join(t_string, str(obj_counter) + "img.png"), img)

            state = 3

        elif state == 3:
            # Analyze affordance

            print("Segmenting affordance maps")
            aff_client = AffordanceClient()

            aff_client.start(GPU=True)
            _ = aff_client.run(img, CONF_THRESHOLD = 0.5)

            masks, labels, scores, bboxs = aff_client.getAffordanceResult()
            masks = aff_client.processMasks(masks, conf_threshold = 0, erode_kernel=(1,1))

            if args.save:
                np.save(os.path.join(t_string, str(obj_counter) + "masks.npy"), masks)
                np.save(os.path.join(t_string, str(obj_counter) + "labels.npy"), labels)
                np.save(os.path.join(t_string, str(obj_counter) + "scores.npy"), scores)
                np.save(os.path.join(t_string, str(obj_counter) + "bboxs.npy"), bboxs)

            print("Found the following objects, waiting for command: ")
            for label in labels:
                print(aff_client.OBJ_CLASSES[label])

            #cv2.imshow("masks", visualizeBBoxInRGB(visualizeMasksInRGB(img, masks), labels, bboxs, scores))
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            state = 4


        elif state == 4:

            req_obj_id = obj_ids[obj_counter]
            while True:
                rate.sleep()
                if req_obj_id != -1:
                    break
            state = 5

        elif state == 5:
            # Check user input

            obj_counter += 1

            try:
                obj_inst = np.where(labels == req_obj_id)[0][0]
                print("Attempting to pick up: ", aff_client.OBJ_CLASSES[labels[obj_inst]])
                state = 6
            except:
                print("Did not find requested object")
                req_obj_id = -1
                req_obj_id = -1
                state = 2

        elif state == 6:
            # post process affordance segmentation maps

            obj_inst_masks = masks[obj_inst]
            obj_inst_label = labels[obj_inst]
            obj_inst_bbox = bboxs[obj_inst]

            if args.save:
                np.save(os.path.join(t_string, str(obj_counter) + str(req_obj_id) + "_premasks.npy"), obj_inst_masks)
                np.save(os.path.join(t_string, str(obj_counter) + str(req_obj_id) + "_label.npy"), obj_inst_label)
                np.save(os.path.join(t_string, str(obj_counter) + str(req_obj_id) + "_bboxs.npy"), obj_inst_bbox)

            # Post process affordance predictions and compute point cloud affordance mask

            affordances_in_object = getPredictedAffordances(masks = obj_inst_masks, bbox = obj_inst_bbox)
            print("predicted affordances", affordances_in_object)

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

            obj_inst_masks = removeOverlapMask(masks = obj_inst_masks)
            #cv2.imshow("masks", visualizeMasksInRGB(img, obj_inst_masks))
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            if args.save:
                np.save(os.path.join(t_string, str(obj_counter) + str(req_obj_id) + "_postmasks.npy"), obj_inst_masks)

            affordances_in_object = getPredictedAffordances(masks = obj_inst_masks, bbox = obj_inst_bbox)

            print("Predicted affordances in object: ")
            for aff in affordances_in_object:
                print(aff_client.labelsToNames[aff])

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

            transform.visualizeTransform(tf_msg, "world_to_camera")


            pcd.transform(T)
            points = np.asanyarray(pcd.points)
            pcd_affordance = getObjectAffordancePointCloud(pcd, obj_inst_masks, uvs = cloud_uv)

            # Compute a downsampled version of the point cloud for collision checking
            # downsampling speeds up computation
            #pcd_downsample = pcd.voxel_down_sample(voxel_size=0.005)
            pcd_downsample = pcd.voxel_down_sample(voxel_size=0.01)

            state = 8

        elif state == 8:

            # Select affordance mask to compute grasps for
            observed_affordances = getPredictedAffordances(obj_inst_masks)
            functionalLabels = [2, 3, 4, 5, 6, 7, 8, 9]

            success = []
            sampled_grasp_points = []
            for observed_affordance in observed_affordances:
                if observed_affordance in aff_client.functionalLabels:
                    local_success, local_sampled_grasp_points = getPointCloudAffordanceMask(affordance_id = observed_affordance,
                                                    points = points, uvs = cloud_uv, masks = obj_inst_masks)
                    success.append(local_success)

                    if len(sampled_grasp_points) == 0:
                        sampled_grasp_points = local_sampled_grasp_points
                    else:
                        sampled_grasp_points = np.vstack((sampled_grasp_points, local_sampled_grasp_points))

            if args.save:
                o3d.io.write_point_cloud(os.path.join(t_string, str(obj_counter) + "pcd_world.ply"), pcd)
                o3d.io.write_point_cloud(os.path.join(t_string, str(obj_counter) + "pcd_affordance.ply"), pcd_affordance)

            if True in success:

                # computing goal pose of object in world frame and
                # current pose of object in world frame

                rotClient = OrientationClient()
                rotClient.setSettings(1)
                current_orientation, current_position, goal_orientation_giver = rotClient.getOrientation(pcd_affordance) # we discard translation

                curr_rot_quat_world = R.from_matrix(current_orientation).as_quat()
                curr_pose_world = np.hstack((current_position.flatten(), curr_rot_quat_world))
                transform.visualizeTransform(transform.poseToTransform(curr_pose_world), "object_current_pose")
                rotated_coordinate_frame = visualizeFrameMesh(current_position, current_orientation)

                #o3d.visualization.draw_geometries([pcd, rotated_coordinate_frame])

                if args.save:
                    np.save(os.path.join(t_string, str(obj_counter) + "current_pose_world.npy"), curr_pose_world)

                loc_client = LocationClient()
                goal_location_giver_original = loc_client.getLocation().flatten() # in givers frame

                _, _, rotMatGiver2World = transform.getTransform("giver", "world")

                # run the grasp algorithm
                grasp_client = GraspingGeneratorClient()
                grasp_client.setSettings(0.05, -1.0, 1.0, # azimuth
                                        0.1, -0., 0., # polar
                                        0.0025, -0.005, 0.05) # depth
                grasps = grasp_client.run(sampled_grasp_points, pcd_downsample,
                                            "world", req_obj_id, -1, obj_inst)
                grasps.sortByScore()

                for count_grasp, grasp in enumerate(grasps):

                    print("=========================================")
                    print("Computing for grasp num: ", count_grasp + 1, " / ", len(grasps))

                    valid_waypoints = [0, 0, 0]
                    waypoint = computeWaypoint(grasp, offset = 0.1)
                    waypoint_msg = waypoint.toPoseMsg()
                    pub_waypoint.publish(waypoint.toPoseStampedMsg())
                    valid_waypoint, state_waypoint = moveit.getInverseKinematicsSolution(state_ready, waypoint_msg)
                    if valid_waypoint:
                        valid_waypoints = [1, 0, 0]
                        grasp_msg = grasp.toPoseMsg()
                        valid_grasp, state_grasp = moveit.getInverseKinematicsSolution(state_waypoint.solution, grasp_msg)

                        if valid_grasp:
                            valid_waypoints = [1, 1, 0]
                            pub_waypoint.publish(waypoint.toPoseStampedMsg())
                            pub_grasp.publish(grasp.toPoseStampedMsg())

                            rot_range = 7
                            x_range = 5
                            for x_pos in range(x_range):
                                for rot_num in range(rot_range):

                                    if state != 2:

                                        # tools rotate around giver frame's x axis
                                        # contain objects rotate around giver frame's z-axis

                                        if obj_inst_label in [1, 2, 3, 4, 5, 6, 7, 10, 13, 14, 15, 16, 17, 17, 18, 19, 20]:
                                            rotation = R.from_euler("xyz", [0, 0, rot_num * (2 * math.pi / rot_range)]).as_matrix()
                                            rotation = R.from_euler("XYZ", [rot_num * (2 * math.pi / rot_range), 0, 0]).as_matrix()
                                        else:
                                            rotation = R.from_euler("xyz", [0, 0, rot_num * (2 * math.pi / rot_range)]).as_matrix()
                                        goal_orientation_giver = np.matmul(rotation, goal_orientation_giver)
                                        goal_orientation_world = np.matmul(rotMatGiver2World, goal_orientation_giver)
                                        goal_rot_quat = R.from_matrix(goal_orientation_world).as_quat()

                                        goal_location_giver = np.zeros(3)
                                        goal_location_giver[0] = min(1.2, max(0.7 ,goal_location_giver_original[0]/2))
                                        goal_location_giver[0] += 0.2 - (0.4 / x_range) * x_pos
                                        goal_location_giver = np.reshape(goal_location_giver, (3, 1))
                                        goal_location = transform.transformToFrame(goal_location_giver, "world", "giver")
                                        goal_location = np.array([goal_location.pose.position.x, goal_location.pose.position.y, goal_location.pose.position.z])
                                        goal_location[2] = 1.2

                                        print(x_pos, rot_num)

                                        goal_pose_world = np.hstack((goal_location.flatten(), goal_rot_quat))
                                        #transform.visualizeTransform(transform.poseToTransform(goal_pose_world), "object_goal_pose")

                                        # Compute the homegenous 4x4 transformation matrices

                                        world_grasp_T = transform.poseStampedToMatrix(grasp.toPoseStampedMsg()) # grasp_pose_world
                                        world_centroid_T = transform.poseToMatrix(curr_pose_world)
                                        world_centroid_T_goal = transform.poseToMatrix(goal_pose_world)

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


                                        valid_handover, state_handover = moveit.getInverseKinematicsSolution(state_ready, ee_pose)

                                        #print("Executing trajectory")
                                        if valid_handover:

                                            if args.save:
                                                np.save(os.path.join(t_string, str(obj_counter) + "grasp.npy"), grasp.toPoseArray())

                                            transform.visualizeTransform(ee_tf, "goal_EE_pose")
                                            valid_waypoints = [1, 1, 1]
                                            print("Valid waypoints ", valid_waypoints)

                                            print("Moving to waypoint...")
                                            result = rob9Utils.iiwa.execute_ptp(moveit.getJointPositionAtNamed("ready").joint_position.data)
                                            result = rob9Utils.iiwa.execute_ptp(state_waypoint.solution.joint_state.position[0:7])
                                            rospy.sleep(1)

                                            print("Moving to grasp pose...")
                                            result = rob9Utils.iiwa.execute_ptp(state_grasp.solution.joint_state.position[0:7])
                                            rospy.sleep(1)

                                            gripper_pub.publish(close_gripper_msg)
                                            rospy.sleep(1)

                                            print("I have grasped!")
                                            print("Moving to ready...")
                                            result = rob9Utils.iiwa.execute_ptp(state_waypoint.solution.joint_state.position[0:7])
                                            result = rob9Utils.iiwa.execute_ptp(moveit.getJointPositionAtNamed("ready").joint_position.data)

                                            # Execute plan to handover pose
                                            result = rob9Utils.iiwa.execute_ptp(state_handover.solution.joint_state.position[0:7])
                                            rospy.sleep(2)

                                            gripper_pub.publish(open_gripper_msg)
                                            rospy.sleep(1)
                                            result = rob9Utils.iiwa.execute_ptp(moveit.getJointPositionAtNamed("ready").joint_position.data)
                                            result = rob9Utils.iiwa.execute_ptp(moveit.getJointPositionAtNamed("camera_ready_3").joint_position.data)
                                            print("Motion complete")
                                            state = 2
                                            break

                    print("Valid waypoints ", valid_waypoints)
            state = 2

        elif state == 9:
            # restart
            req_obj_id = -1

            state = 1

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
