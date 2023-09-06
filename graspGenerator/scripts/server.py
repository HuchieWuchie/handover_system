#!/usr/bin/env python3

import os
import sys
import rospy
import numpy as np
import open3d as o3d
import cv2
import math
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors

from nav_msgs.msg import Path
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation
from std_msgs.msg import Header, Float32

from grasp_generator.srv import *
from rob9.msg import *
from rob9.srv import *
from rob9Utils.visualize import create_mesh_box, createGripper, visualizeGripper, visualizeFrameMesh
from grasp_service.client import GraspingGeneratorClient
from cameraService.cameraClient import CameraClient
from scipy.spatial import distance
#from rob9Utils.graspGroup import GraspGroup as rob9GraspGroup
#from rob9Utils.grasp import Grasp as rob9Grasp

class GraspServer(object):
    """docstring for GraspServer."""

    def __init__(self):

        print('Starting...')
        rospy.init_node('grasp_generator', anonymous=True)

        self.rate = rospy.Rate(5)

        # Default values
        self.azimuth_step_size = 0.025
        self.azimuth_min = 0
        self.azimuth_max = 0.6

        self.polar_step_size = 0.05
        self.polar_min = 0.0
        self.polar_max = 0.2

        self.depth_step_size = 0.01 # m
        self.depth_min = 0
        self.depth_max = 0.03

        self.serviceRun = rospy.Service("grasp_generator/result", runGraspingSrv, self.run)
        self.serviceSetSettings = rospy.Service("grasp_generator/set_settings", setSettingsGraspingSrv, self.setSettings)

    def setSettings(self, msg):

        self.azimuth_step_size = msg.azimuth_step_size.data
        self.azimuth_min = msg.azimuth_min.data
        self.azimuth_max = msg.azimuth_max.data

        self.polar_step_size = msg.polar_step_size.data
        self.polar_min = msg.polar_min.data
        self.polar_max = msg.polar_max.data

        self.depth_step_size = msg.depth_step_size.data # m
        self.depth_min = msg.depth_min.data
        self.depth_max = msg.depth_max.data

        print("Updated settings")

        return setSettingsGraspingSrvResponse()

    def run(self, msg):

        cam_client = CameraClient()


        print("Computing...")
        sampled_grasp_points, _ = cam_client.unpackPCD(msg.grasp_points, None)
        pcd_env_points, _ = cam_client.unpackPCD(msg.pcd_env, None)
        frame_id = msg.frame_id.data
        tool_id = msg.tool_id.data
        affordance_id = msg.affordance_id.data
        obj_inst = msg.object_instance.data

        pcd_downsample = o3d.geometry.PointCloud()
        pcd_downsample.points = o3d.utility.Vector3dVector(pcd_env_points)

        sampled_grasps = o3d.geometry.PointCloud()
        sampled_grasps.points = o3d.utility.Vector3dVector(sampled_grasp_points)
        sampled_grasps = sampled_grasps.voxel_down_sample(voxel_size=0.02)
        sampled_grasp_points = np.asanyarray(sampled_grasps.points)

        polar_values = np.arange(self.polar_min, self.polar_max + self.polar_step_size,
                                    self.polar_step_size)
        azimuth_values = np.arange(self.azimuth_min, self.azimuth_max + self.azimuth_step_size,
                                    self.azimuth_step_size)
        depth_values = np.arange(self.depth_min, self.depth_max + self.depth_step_size,
                                    self.depth_step_size)

        centroid = sampled_grasps.get_center()
        bounds = sampled_grasps.get_max_bound() - sampled_grasps.get_min_bound()

        poses, scores = [], []
        for grasp_count, s_grasp in enumerate(np.asanyarray(sampled_grasp_points)):


            local_points = np.asanyarray(pcd_downsample.points)
            idx_min_x = local_points[:,0] > (s_grasp[0] - 0.1)
            local_points = local_points[idx_min_x]
            idx_max_x = local_points[:,0] < (s_grasp[0] + 0.1)
            local_points = local_points[idx_max_x]

            idx_min_y = local_points[:,1] > (s_grasp[1] - 0.1)
            local_points = local_points[idx_min_y]
            idx_max_y = local_points[:,1] < (s_grasp[1] + 0.1)
            local_points = local_points[idx_max_y]

            idx_min_z = local_points[:,2] > (s_grasp[2] - 0.1)
            local_points = local_points[idx_min_z]
            idx_max_z = local_points[:,2] < (s_grasp[2] + 0.1)
            local_points = local_points[idx_max_z]

            blob_matrix = np.zeros((polar_values.shape[0],
                                    azimuth_values.shape[0])).astype(np.uint8)

            sampled_grasps_without_current = []
            for g_count, g in enumerate(sampled_grasp_points):
                if g_count != grasp_count:
                    sampled_grasps_without_current.append(g)
            sampled_grasps_without_current = np.array(sampled_grasps_without_current)
            neigh= NearestNeighbors(n_neighbors=1)
            neigh.fit(sampled_grasps_without_current)

            largest_dist = 0
            best_pol_val = 0
            best_azi_val = 0

            for y_count, polar_value in enumerate(polar_values):
                for x_count, azimuth_value in enumerate(azimuth_values):

                    # compute translation
                    translation = s_grasp.copy()
                    translation[2] = s_grasp[2]# - depth_value

                    # compute orientation

                    ee_rotation = np.array([math.pi + (math.pi * polar_value), 0, (math.pi *azimuth_value)]) # franka
                    #ee_rotation = np.array([(math.pi / 2.0) + (math.pi * azimuth_value) + math.pi, 0, (math.pi) + (math.pi * polar_value)])
                    rotEE = R.from_euler('XYZ', ee_rotation)
                    eeRotMat = rotEE.as_matrix()

                    #world_gripper = createGripper(opening = 0.08, translation = np.zeros(3), rotation = np.identity(3))
                    #vis_world_gripper = visualizeGripper(world_gripper)
                    #world_coordinate_frame = visualizeFrameMesh(np.zeros(3),np.identity(3))
                    #o3d.visualization.draw_geometries([pcd_downsample,  vis_world_gripper, world_coordinate_frame])


                    gripper = createGripper(opening = 0.12, translation = translation, rotation = eeRotMat)
                    if self.checkCollisionEnvironment(gripper, local_points) == False:
                        if self.checkCollisionEnvironment(gripper, sampled_grasp_points) == False:
                            #vis_gripper = visualizeGripper(gripper)
                            #gripper_frame = visualizeFrameMesh(translation, eeRotMat)
                            #o3d.visualization.draw_geometries([pcd_downsample,  vis_gripper, gripper_frame])
                            blob_matrix[y_count, x_count] = 255 #outcommented today
                            distance_left_finger, _ = neigh.kneighbors(np.reshape(gripper[1].get_center(), (1, 3)), return_distance = True)
                            distance_right_finger, _ = neigh.kneighbors(np.reshape(gripper[2].get_center(), (1, 3)), return_distance = True)
                            distance_left_finger = distance_left_finger[0][0]
                            distance_right_finger = distance_right_finger[0][0]
                            dist_to_self = min(distance_left_finger, distance_right_finger)

                            if dist_to_self > largest_dist:
                                largest_dist = dist_to_self
                                best_azi_val = azimuth_value
                                best_pol_val = polar_value
                            #print("Colliion self FALSE")

                            #vis_gripper = visualizeGripper(gripper)
                            #gripper_frame = visualizeFrameMesh(translation, eeRotMat)
                            #o3d.visualization.draw_geometries([pcd_downsample,  vis_gripper, gripper_frame])
                        #else:
                        #    print("Collision self TRUE")
                    #else:
                    #    print("Colliion environment TRUE")



            if 255 in np.unique(blob_matrix):
                #best_grasp_idx, score = self.processGrasps(blob_matrix)
                #score = largest_dist

                #polar_value = polar_values[best_grasp_idx[0]]
                #azimuth_value = azimuth_values[best_grasp_idx[1]]

                polar_value = best_pol_val
                azimuth_value = best_azi_val

                ee_rotation = np.array([math.pi + (math.pi * polar_value), 0, (math.pi *azimuth_value)]) # franka
                #ee_rotation = np.array([0, math.pi / 2, math.pi/2])
                #ee_rotation = np.array([0, math.pi / 2, 0])
                #ee_rotation = np.array([(math.pi / 2.0) + (math.pi * azimuth_value) + math.pi, 0, (math.pi) + (math.pi * polar_value)])
                #rotEE = R.from_euler('ZYX', ee_rotation)
                rotEE = R.from_euler('XYZ', ee_rotation)
                eeRotMat = rotEE.as_matrix()

                score = 0

                d_count = 0
                for depth_value in depth_values:

                    translation = s_grasp.copy()
                    translation[2] = translation[2] - depth_value
                    gripper = createGripper(opening = 0.12, translation = translation, rotation = eeRotMat)

                    distance_left_finger, _ = neigh.kneighbors(np.reshape(gripper[1].get_center(), (1, 3)), return_distance = True)
                    distance_right_finger, _ = neigh.kneighbors(np.reshape(gripper[2].get_center(), (1, 3)), return_distance = True)
                    distance_left_finger = distance_left_finger[0][0]
                    distance_right_finger = distance_right_finger[0][0]
                    dist_to_self = min(distance_left_finger, distance_right_finger)
                    score += dist_to_self
                    #vis_gripper = visualizeGripper(gripper)
                    #gripper_frame = visualizeFrameMesh(translation, eeRotMat)
                    #o3d.visualization.draw_geometries([pcd_downsample, vis_gripper, gripper_frame])

                    if self.checkCollisionEnvironment(gripper, local_points) == True:
                        break

                    d_count += 1

                #score = (d_count + 1) / depth_values.shape[0]

                translation = s_grasp.copy()
                translation[2] = s_grasp[2] - depth_values[int(d_count / 2)]
                #score = 1 - distance.euclidean(np.linalg.norm(translation), np.linalg.norm(centroid))
                #translation[2] = s_grasp[2] - depth_values[min(0, d_count-1)]

                print(score)
                #vis_gripper = visualizeGripper(gripper)
                #gripper_frame = visualizeFrameMesh(translation, eeRotMat)
                #o3d.visualization.draw_geometries([pcd_downsample, vis_gripper, gripper_frame])

                quat = rotEE.as_quat()
                poses.append([translation[0], translation[1], translation[2],
                            quat[0], quat[1], quat[2], quat[3]])
                scores.append(score)

        print("Computed grasps, now sending...")

        grasp_client = GraspingGeneratorClient()
        grasp_msg = grasp_client.packGrasps(poses, scores, frame_id, tool_id,
                                            affordance_id, obj_inst)

        response = runGraspingSrvResponse()
        response.grasps = grasp_msg

        return response

    def processGrasps(self, blob_img):


        contours, hierarchy = cv2.findContours(blob_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        hulls = []
        for contour in contours:
            hulls.append(cv2.convexHull(contour, False))

        largest_blob = 0
        idx = [0,0]
        for i in range(len(hulls)):
            im = np.zeros((blob_img.shape[0], blob_img.shape[1]))
            cv2.drawContours(im, hulls, i, 255, -1)
            size = np.count_nonzero(im == 255)
            # find largest axis
            true_idxs = np.where(im == 255)
            y_range = np.max(true_idxs[0]) - np.min(true_idxs[0])
            x_range = np.max(true_idxs[1]) - np.min(true_idxs[1])

            x_vals = np.arange(np.min(true_idxs[1]), np.max(true_idxs[1])) - int(x_range/2)
            y_vals = np.arange(np.min(true_idxs[0]), np.max(true_idxs[0])) - int(y_range/2)

            k_size = 0
            x_start = int(np.median(true_idxs[1]))
            y_start = int(np.median(true_idxs[0]))
            while True:
                local_area = im[y_start - k_size : y_start + k_size + 1,
                                x_start - k_size : x_start + k_size + 1]
                if 255 in np.unique(local_area):
                    idx_y = np.where(local_area == 255)[0][0] + y_start
                    idx_x = np.where(local_area == 255)[1][0] + x_start
                    idx = [idx_y, idx_x]
                    break
                k_size +=1

            if size > largest_blob:
                largest_blob = size

        score = largest_blob / (im.shape[0] * im.shape[1])

        return idx, score

    def insideCubeTest(self, cube, points):
        """
        cube =  numpy array of the shape (8,3) with coordinates in the clockwise order. first the bottom plane is considered then the top one.
        points = array of points with shape (N, 3).
        Returns the indices of the points array which are outside the cube3d
        modified: https://stackoverflow.com/questions/21037241/how-to-determine-a-point-is-inside-or-outside-a-cube
        """

        b1,b2,b4,t1,t3,t4,t2,b3 = cube

        dir1 = (t1-b1)
        size1 = np.linalg.norm(dir1)
        dir1 = dir1 / size1

        dir2 = (b2-b1)
        size2 = np.linalg.norm(dir2)
        dir2 = dir2 / size2

        dir3 = (b4-b1)
        size3 = np.linalg.norm(dir3)
        dir3 = dir3 / size3

        cube3d_center = (b1 + t3)/2.0

        dir_vec = points - cube3d_center

        res1 = np.where( (np.absolute(np.dot(dir_vec, dir1)) * 2) > size1 )[0]
        res2 = np.where( (np.absolute(np.dot(dir_vec, dir2)) * 2) > size2 )[0]
        res3 = np.where( (np.absolute(np.dot(dir_vec, dir3)) * 2) > size3 )[0]

        return list( set().union(res1, res2, res3) )


    def checkCollisionEnvironment(self, gripper, points):


        points_l_finger = np.asanyarray(gripper[1].get_oriented_bounding_box().get_box_points())
        points_r_finger = np.asanyarray(gripper[2].get_oriented_bounding_box().get_box_points())
        points_chasis = np.asanyarray(gripper[0].get_oriented_bounding_box().get_box_points())

        # check collision left_finger
        points_outside = self.insideCubeTest(points_l_finger,
                                        points)
        if len(points_outside) != points.shape[0]:
            return True


        # check collision right_finger
        points_outside = self.insideCubeTest(points_r_finger,
                                        points)
        if len(points_outside) != points.shape[0]:
            return True

        # check collisoin chasis
        points_outside = self.insideCubeTest(points_chasis,
                                        points)
        if len(points_outside) != points.shape[0]:
            return True
        return False

if __name__ == "__main__":

    graspGenerator = GraspServer()
    print("Grasp generator is ready")

    while not rospy.is_shutdown():
        rospy.spin()
