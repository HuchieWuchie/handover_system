#!/usr/bin/env python3

import os
import rospy
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R
import random
import math
import time
import sys, signal
import cv2
import scipy.optimize

from geometry_msgs.msg import Pose
from std_msgs.msg import Header, Float32

from orientation_service.srv import runOrientationSrv, runOrientationSrvResponse
from orientation_service.srv import setSettingsOrientationSrv, setSettingsOrientationSrvResponse

from rob9Utils.affordancetools import getAffordancePointCloudBasedOnVariance, getPredictedAffordances, getAffordanceColors, getAffordanceContours, getObjectAffordancePointCloud, getAffordanceBoundingBoxes, getPredictedAffordancesInPointCloud
from rob9Utils.utils import erodeMask, keepLargestContour, convexHullFromContours, maskFromConvexHull, thresholdMaskBySize, removeOverlapMask

from cameraService.cameraClient import CameraClient
from affordanceService.client import AffordanceClient
from orientationService.client import OrientationClient

def signal_handler(signal, frame):
    print("Shutting down program.")
    sys.exit()

signal.signal(signal.SIGINT, signal_handler)

class OrientationServer(object):
    """docstring for OrientationServer."""

    def __init__(self):

        print('Starting...')
        #rospy.init_node('orientation_service', anonymous=True)
        self.serviceRun = rospy.Service("/computation/handover_orientation/get", runOrientationSrv, self.run)
        self.serviceRun = rospy.Service("/computation/handover_orientation/set_settings", setSettingsOrientationSrv, self.setSettings)

        self.renderer_width = 640
        self.renderer_height = 360
        self.renderer_fx = 317.373
        self.renderer_fy = 317.373
        self.renderer_cx = 323.836
        self.renderer_cy = 178.645

        self.intrinsics = o3d.camera.PinholeCameraIntrinsic(self.renderer_width, self.renderer_height,
                            self.renderer_fx, self.renderer_fy, self.renderer_cx, self.renderer_cy)
        self.extrinsics = np.eye(4)
        self.extrinsics[3,3] = 1.5

        self.renderer = o3d.visualization.rendering.OffscreenRenderer(self.renderer_width,
                        self.renderer_height)
        self.renderer.scene.set_background(np.array([0,0,0,0]))
        self.renderer.setup_camera(self.intrinsics, self.extrinsics)

        self.material = o3d.visualization.rendering.MaterialRecord()
        self.material.shader = 'defaultUnlit'
        self.material.point_size = 15

        self.rate = rospy.Rate(5)

        root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                    "sampled_object_point_clouds")
        self.pcd_paths = []
        self.pcd_paths.append(os.path.join(root_dir, "14_spatula_14.txt"))
        self.pcd_paths.append(os.path.join(root_dir, "1_knife_8.txt"))
        self.pcd_paths.append(os.path.join(root_dir, "15_hammer_13.txt"))
        self.pcd_paths.append(os.path.join(root_dir, "5_scoop_5.txt"))
        self.pcd_paths.append(os.path.join(root_dir, "10_ladle_7.txt"))
        self.pcd_paths.append(os.path.join(root_dir, "9_cup_16.txt"))
        self.pcd_paths.append(os.path.join(root_dir, "11_mug_1.txt"))
        self.pcd_paths.append(os.path.join(root_dir, "18_bottle_3.txt"))

        self.mean_quat = []
        self.mean_quat.append([ 0.17858507, -0.72325377,  0.14493605,  0.65115659]) # spatula
        self.mean_quat.append([ 0.05262527, -0.72957124,  0.15906051,  0.66306572]) # knife
        self.mean_quat.append([ 0.04510172,  0.6002882,  -0.32359202, -0.73000556]) # hammer
        self.mean_quat.append([ 0.09140731, -0.71904327,  0.15291612,  0.67174261]) # scoop
        self.mean_quat.append([-0.03664714,  0.72422009, -0.08418687, -0.68342872]) # ladle
        self.mean_quat.append([-0.0985996,   0.70367063 ,-0.18686055, -0.67838698]) # cup
        self.mean_quat.append([ 0.25167641,  0.69930462,  0.32368484, -0.58554261]) # mug
        self.mean_quat.append([-0.66268984, -0.04537715, -0.68318855,  0.30337519]) # bottle

        self.nn_classifier = self.createNNClassifier()

        self.method = 0 # observation based

    def setSettings(self, msg):

        int_to_method = {0: "observation-based", 1: "rule-based"}

        self.method = msg.method.data

        print("Changed handover observatoin computation method to: ", int_to_method[self.method])
        return setSettingsOrientationSrvResponse()

    def createSourceWGrasp(self):

        points = []
        colors = []

        for i in range(15):
            x = -0.4 + i * 0.05
            length = 0.08 + 0.01*i

            for j in range(16):
                z = (length * math.cos(math.pi * (j / 8)) ) #- (length * math.sin(math.pi * j))
                y = (length * math.sin(math.pi * (j / 8)))
                points.append([x, y, z])
                colors.append((255, 0, 255))

        points = np.array(points)
        colors = np.array(colors)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd, np.identity(3)

    def createSourceGrasp(self):

        label_colors = {1: (0, 0, 255), # grasp
        2: (0, 255, 0), # cut
        3: (123, 255, 123), # scoop
        4: (255, 0, 0), # contain
        5: (255, 255, 0), # pound
        6: (255, 255, 255), # support
        7: (255, 0, 255)} # wrap-grasp

        # grasp
        points = []
        colors = []

        for i in range(15):
            x = 0.5 + (-i * 0.02)
            h, d = 0.02, 0.02

            points.append([x, h, d])
            points.append([x, -h, d])
            points.append([x, h, -d])
            points.append([x, -h, -d])

            colors.append((0, 0, 255))
            colors.append((0, 0, 255))
            colors.append((0, 0, 255))
            colors.append((0, 0, 255))

        for i in range(2,7):
            for j in range(5):

                x = -0.2 - (j * 0.02)
                h, d = 0.02, 0.02

                points.append([x, h, d])
                points.append([x, -h, d])
                points.append([x, h, -d])
                points.append([x, -h, -d])

                colors.append(label_colors[i])
                colors.append(label_colors[i])
                colors.append(label_colors[i])
                colors.append(label_colors[i])

        points = np.array(points)
        colors = np.array(colors)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd, np.identity(3)

    def preparePointCloudForRenderer(self, pcd):
        """ Scales and translates the input point cloud to a size that fits with
            the open3d offscreen renderer, also normalizes colors.

            Input:
            pcd             - o3d.geometry.PointCloud()

            Output:
            pcd             - o3d.geometry.PointCloud(), scaled and color normalized.
        """

        if np.max(np.asanyarray(pcd.colors) > 1):
            pcd.colors = o3d.utility.Vector3dVector(np.asanyarray(pcd.colors) / 255)

        bounds = pcd.get_max_bound() - pcd.get_min_bound()
        scale = 0.2 / np.max(bounds)
        pcd.points = o3d.utility.Vector3dVector(np.asanyarray(pcd.points) * scale)
        bounds = pcd.get_max_bound() - pcd.get_min_bound()

        pcd.translate(-pcd.get_center())
        pcd.translate([0, 0, bounds[2] + 0.15])

        return pcd

    def postProcessSingleViewPointCloud(self, pcd):
        """ Post processing by changing color values and cropping away background

            Input:
            pcd             - o3d.geometry.PointCloud(), rendered single view point cloud

            Output:
            pcd             - o3d.geometry.PointCloud(), post processesing applied
        """

        bbox_min = np.array([-10,-10, 0])
        bbox_max = np.array([10,10, 0.9])
        bbox = o3d.geometry.AxisAlignedBoundingBox(bbox_min, bbox_max)
        pcd = pcd.crop(bbox)
        pcd.translate(-pcd.get_center())

        colors = np.asanyarray(pcd.colors)
        if np.max(colors) <= 1.0:
            colors = colors * 255

        colors[colors < 30] = 0
        colors[colors > 220] = 255

        idx = np.where(np.logical_and(colors > 95, colors < 150))
        colors[idx[0], idx[1]] = 123

        idx = np.where(np.logical_and(colors >= 30, colors <= 95))
        colors[idx[0], idx[1]] = 0

        idx = np.where(np.logical_and(colors >= 150, colors <= 220))
        colors[idx[0], idx[1]] = 0

        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def scalePcdToTargetPcd(self, source_pcd, target_pcd):
        """ Scales the size of source_pcd to roughly the size of target_pcd

            Input:
            source_pcd      - o3d.geometry.PointCloud()
            target_pcd      - o3d.geometry.PointCloud()

            Output:
            pcd_scaled      - source_pcd scaled
        """

        source_bounds = source_pcd.get_max_bound() - source_pcd.get_min_bound()
        target_bounds = target_pcd.get_max_bound() - target_pcd.get_min_bound()
        scale = np.max(target_bounds) / np.max(source_bounds)

        pcd_scaled = o3d.geometry.PointCloud(source_pcd)
        pcd_scaled.points = o3d.utility.Vector3dVector(np.asanyarray(pcd_scaled.points) * scale)

        return pcd_scaled

    def findBestTransform(self, target_pcd, source_pcd, x_range = 5,
                        y_range = 5, z_range = 5):
        global depth_image, rgb_image, capture
        """ Samples source_pcd from single view with different rotations applied and
            finds the best transform to target_pcd using affordance ICP

            Input:
            taget_pcd       - o3d.geometry.PointCloud()
            source_pcd      - o3d.geometry.PointCloud()

            Output:
            best_T               - (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
        """

        ts = time.time()

        best_T = np.eye(4)
        best_score = 10000000000000
        transformation = np.eye(4)

        target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
        target_pcd.normalize_normals()

        #source_pcd_downscaled = self.preparePointCloudForRenderer(source_pcd)

        depth_image, rgb_image = None, None
        for x in range(x_range):
            for y in range(y_range):
                for z in range(z_range):

                    xr = (2*math.pi / x_range) * x
                    yr = (2*math.pi / y_range) * y
                    zr = (2*math.pi / z_range) * z

                    rot_mat = R.from_euler("xyz", [xr, yr, zr]).as_matrix()
                    pcd_rotated = o3d.geometry.PointCloud(source_pcd)
                    pcd_rotated.rotate(rot_mat)

                    self.renderer.scene.clear_geometry()
                    self.renderer.scene.add_geometry("pcd", pcd_rotated, self.material)

                    capture = True
                    waiting = True
                    while waiting:
                        if capture:
                            time.sleep(0.1)
                        else:
                            waiting = False

                    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, depth_image,
                                                                            depth_scale = 1,
                                                                            convert_rgb_to_intensity = False)
                    #depth_image, rgb_image = None, None
                    pcd_single_view = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsics,
                                                                                    project_valid_depth_only = True)
                    pcd_single_view = self.postProcessSingleViewPointCloud(pcd_single_view)
                    pcd_single_view = self.scalePcdToTargetPcd(pcd_single_view, target_pcd)

                    pcd_single_view.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
                    pcd_single_view.orient_normals_towards_camera_location
                    pcd_single_view.normalize_normals()

                    source_normals = np.asanyarray(pcd_single_view.normals)
                    target_normals = np.asanyarray(target_pcd.normals)

                    #o3d.visualization.draw_geometries([target_pcd, pcd_single_view])
                    """

                    source_pcd_box = getAffordancePointCloudBasedOnVariance(pcd_single_view)
                    target_pcd_box = getAffordancePointCloudBasedOnVariance(target_pcd)

                    source_affordances, _ = getPredictedAffordancesInPointCloud(source_pcd_box)
                    source_affordances[0] = 0
                    target_affordances, _ = getPredictedAffordancesInPointCloud(target_pcd_box)

                    if np.array_equal(source_affordances, target_affordances):

                        T, distances, iterations = self.icp(source_points = np.asanyarray(source_pcd_box.points),
                                                        source_colors = np.asanyarray(source_pcd_box.colors),
                                                        target_points = np.asanyarray(target_pcd_box.points),
                                                        target_colors = np.asanyarray(target_pcd_box.colors),
                                                        tolerance=0.00001)

                        score = np.mean(distances)

                    else:
                        score = 9999
                        T = np.eye(4)
                    """

                    source_affordances, _ = getPredictedAffordancesInPointCloud(pcd_single_view)
                    source_affordances[0] = 0
                    target_affordances, _ = getPredictedAffordancesInPointCloud(target_pcd)

                    if np.array_equal(source_affordances, target_affordances):

                        T, distances, iterations = self.icpWithNormals(source_points = np.asanyarray(pcd_single_view.points),
                                                        source_colors = np.asanyarray(pcd_single_view.colors),
                                                        source_normals = np.asanyarray(pcd_single_view.normals),
                                                        target_points = np.asanyarray(target_pcd.points),
                                                        target_colors = np.asanyarray(target_pcd.colors),
                                                        target_normals = np.asanyarray(target_pcd.normals),
                                                        tolerance=0.00001)

                        score = np.mean(distances)

                    else:
                        score = 9999
                        T = np.eye(4)


                    print(best_score)

                    if score < best_score:
                        pcd_single_view.transform(T)
                        transformation[:3, :3] = rot_mat
                        best_score = score
                        best_T = np.matmul(T, transformation)
                        best_T[0, 3] = pcd_single_view.get_center()[0]
                        best_T[1, 3] = pcd_single_view.get_center()[1]
                        best_T[2, 3] = pcd_single_view.get_center()[2]

                        o3d.visualization.draw_geometries([target_pcd, pcd_single_view])

                    if best_score < 0.1:
                        break

        te = time.time()
        print("Found transformation in: ", te -ts, " s")

        return best_T




    def methodRule(self, pcd_affordance):
        """ Input:
            pcd_affordance  - o3d.geometry.PointCloud(), point cloud of object
                              where each point's color is their respective
                              affordance
            Output:
            T               - np.array(), shape (4,4) homogeneous transformation
                              matrix describing current pose of object
            G               - goal orientation
        """
        feature_vector = self.computeFeatureVector(pcd_affordance)
        if feature_vector[0][7] == 1:
            source_pcd, goal_orientation = self.createSourceWGrasp()
        else:
            source_pcd, goal_orientation = self.createSourceGrasp()

        target_bounds = pcd_affordance.get_max_bound() - pcd_affordance.get_min_bound()
        source_bounds = source_pcd.get_max_bound() - source_pcd.get_min_bound()
        scale = np.max(target_bounds) / np.max(source_bounds)

        source_pcd_points = np.asarray(source_pcd.points)
        source_pcd_points = source_pcd_points * scale
        source_pcd.points = o3d.utility.Vector3dVector(source_pcd_points)

        T, distances, iterations = self.icp(source_points = np.asanyarray(source_pcd.points),
                                        source_colors = np.asanyarray(source_pcd.colors),
                                        target_points = np.asanyarray(pcd_affordance.points),
                                        target_colors = np.asanyarray(pcd_affordance.colors),
                                        tolerance=0.00001)

        source_pcd.transform(T)
        return T, np.identity(3)

        pass

    def methodObservation(self, pcd_affordance):
        """ Input:
            pcd_affordance  - o3d.geometry.PointCloud(), point cloud of object
                              where each point's color is their respective
                              affordance
            Output:
            T               - np.array(), shape (4,4) homogeneous transformation
                              matrix describing current pose of object
            G               - goal orientation
        """

        feature_vector = self.computeFeatureVector(pcd_affordance)
        dist, predictions = self.nn_classifier.kneighbors(feature_vector, n_neighbors = 1)
        source_pcd = self.getSourcePointCloud(prediction = predictions[0][0])

        target_bounds = pcd_affordance.get_max_bound() - pcd_affordance.get_min_bound()
        source_bounds = source_pcd.get_max_bound() - source_pcd.get_min_bound()
        scale = np.max(target_bounds) / np.max(source_bounds)

        source_pcd_points = np.asarray(source_pcd.points)
        source_pcd_points = source_pcd_points * scale
        source_pcd.points = o3d.utility.Vector3dVector(source_pcd_points)

        T, distances, iterations = self.icp(source_points = np.asanyarray(source_pcd.points),
                                        source_colors = np.asanyarray(source_pcd.colors),
                                        target_points = np.asanyarray(pcd_affordance.points),
                                        target_colors = np.asanyarray(pcd_affordance.colors),
                                        tolerance=0.00001)


        """

        target_bounds = pcd_affordance.get_max_bound() - pcd_affordance.get_min_bound()
        source_bounds = source_pcd.get_max_bound() - source_pcd.get_min_bound()
        scale = np.max(target_bounds) / np.max(source_bounds)

        source_pcd_points = np.asarray(source_pcd.points)
        source_pcd_points = source_pcd_points * scale
        source_pcd.points = o3d.utility.Vector3dVector(source_pcd_points)

        source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
        source_pcd.normalize_normals()

        pcd_affordance.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
        pcd_affordance.normalize_normals()

        T, distances, iterations = self.icpWithNormals(source_points = np.asanyarray(source_pcd.points),
                                        source_colors = np.asanyarray(source_pcd.colors),
                                        source_normals = np.asanyarray(source_pcd.normals),
                                        target_points = np.asanyarray(pcd_affordance.points),
                                        target_colors = np.asanyarray(pcd_affordance.colors),
                                        target_normals = np.asanyarray(pcd_affordance.normals),
                                        tolerance=0.00001)
        """
        #o3d.visualization.draw_geometries([pcd_affordance, source_pcd.transform(T)])
        return T, self.getGoalOrientation(predictions[0][0])

    def get_random_quaternion(self):
        q = np.zeros(4)
        q[0] = np.random.uniform(low=-1.0, high=1.0) #q_x
        q[1] = np.random.uniform(low=-1.0, high=1.0) #q_y
        q[2] = np.random.uniform(low=-1.0, high=1.0) #q_z
        q[3] = np.random.uniform(low=-1.0, high=1.0) #q_w
        # normalize quaternion MAY BE UNNECESSARY IF YOU USE quaternion_from_euler()
        q_mag = np.linalg.norm(q)
        q = q/q_mag
        return q

    def objective_function_with_affordances(self, solution, observation_xyz, observation_colors, source_xyz, source_colors):
        #_, cols = observation.shape
        #distances = np.zeros(cols)

        # center on 0,0,0
        observation_xyz = observation_xyz - np.mean(observation_xyz, axis = 0)
        source_xyz = source_xyz - np.mean(source_xyz, axis = 0)
        #rotated_source = self.rotate_points(solution, source)

        quat = solution
        quat = quat / np.linalg.norm(quat)
        rot_mat = R.from_quat(quat).as_matrix()

        rotated_source = o3d.geometry.PointCloud()
        rotated_source.points = o3d.utility.Vector3dVector(source_xyz)
        rotated_source.colors = o3d.utility.Vector3dVector(source_colors)
        rotated_source.rotate(rot_mat)

        rotated_source = np.hstack((np.asanyarray(rotated_source.points), np.asanyarray(rotated_source.colors)))

        neigh= NearestNeighbors(n_neighbors=1)
        neigh.fit(np.hstack((observation_xyz, observation_colors)))
        distances, _ = neigh.kneighbors(rotated_source, return_distance = True)

        return np.sum(distances)


    def objective_function(self, solution, observation, source):
        #_, cols = observation.shape
        #distances = np.zeros(cols)
        # center on 0,0,0
        observation = observation - np.mean(observation, axis = 0)
        source = source - np.mean(source, axis = 0)
        #rotated_source = self.rotate_points(solution, source)

        quat = solution
        quat = quat / np.linalg.norm(quat)
        rot_mat = R.from_quat(quat).as_matrix()

        rotated_source = o3d.geometry.PointCloud()
        rotated_source.points = o3d.utility.Vector3dVector(source)
        rotated_source.rotate(rot_mat)

        rotated_source = np.asanyarray(rotated_source.points)

        neigh= NearestNeighbors(n_neighbors=1)
        neigh.fit(observation)
        distances, _ = neigh.kneighbors(rotated_source, return_distance = True)
        #print(rotated_source)
        #print(rotated_source.shape)
        #INSERT NEAREST NEIGHBOUR
        #distances = get_distances(observation, rotated_source)
        return np.sum(distances)

    def rotate_points(self, quat, points):
        rows, cols = points.shape
        rot = R.from_quat(quat).as_matrix()
        rotated_pc = np.zeros((rows, 3))
        for i in range(rows):
            rot_point = np.zeros((1,3))
            rot_point = np.matmul(points[i, :], rot)
            rotated_pc[i, :] = rot_point

        return rotated_pc

    def methodObservationQuat(self, pcd_affordance):
        global depth_image, rgb_image, capture
        feature_vector = self.computeFeatureVector(pcd_affordance)
        dist, predictions = self.nn_classifier.kneighbors(feature_vector, n_neighbors = 1)
        source_pcd = self.getSourcePointCloud(prediction = predictions[0][0])
        """

        source_points = np.asanyarray(source_pcd.points)
        target_points = np.asanyarray(pcd_affordance.points)

        num_source_points = source_points.shape[0]
        num_target_points = target_points.shape[0]
        source_pcd = source_pcd.random_down_sample(num_target_points / num_source_points)
        print("=================================")
        print("source_pcd ", source_pcd.points)
        print("pcd_affordance ", pcd_affordance.points)
        print("=================================")

        #source_pcd_for_render = self.preparePointCloudForRenderer(source_pcd)

        #T = self.findBestTransform(pcd_affordance, source_pcd_for_render)

        target_bounds = pcd_affordance.get_max_bound() - pcd_affordance.get_min_bound()
        source_bounds = source_pcd.get_max_bound() - source_pcd.get_min_bound()
        scale = np.max(target_bounds) / np.max(source_bounds)

        source_pcd_points = np.asanyarray(source_pcd.points)
        source_pcd_points = source_pcd_points * scale
        source_pcd.points = o3d.utility.Vector3dVector(source_pcd_points)

        target_points = np.asanyarray(pcd_affordance.points)

        min_sum = None
        min_sum_solution = None

        initial_guess = self.get_random_quaternion()
        xopt = scipy.optimize.minimize(self.objective_function, x0 = initial_guess, method='Nelder-Mead', args=(target_points, source_pcd_points))
        """

        ts = time.time()

        best_T = np.eye(4)
        best_score = 10000000000000
        transformation = np.eye(4)
        source_pcd = self.preparePointCloudForRenderer(source_pcd)

        x_range = 5
        y_range = 5
        z_range = 5

        depth_image, rgb_image = None, None
        count = 1
        for x in range(x_range):
            for y in range(y_range):
                for z in range(z_range):

                    xr = (2*math.pi / x_range) * x
                    yr = (2*math.pi / y_range) * y
                    zr = (2*math.pi / z_range) * z

                    rot_mat = R.from_euler("xyz", [xr, yr, zr]).as_matrix()
                    pcd_rotated = o3d.geometry.PointCloud(source_pcd)
                    pcd_rotated.rotate(rot_mat)

                    self.renderer.scene.clear_geometry()
                    self.renderer.scene.add_geometry("pcd", pcd_rotated, self.material)

                    capture = True
                    waiting = True
                    while waiting:
                        if capture:
                            time.sleep(0.1)
                        else:
                            waiting = False

                    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, depth_image,
                                                                            depth_scale = 1,
                                                                            convert_rgb_to_intensity = False)
                    #depth_image, rgb_image = None, None
                    pcd_single_view = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsics,
                                                                                    project_valid_depth_only = True)
                    pcd_single_view = self.postProcessSingleViewPointCloud(pcd_single_view)
                    pcd_single_view = self.scalePcdToTargetPcd(pcd_single_view, pcd_affordance)

                    num_source_points = np.asanyarray(pcd_single_view.points).shape[0]
                    num_target_points = np.asanyarray(pcd_affordance.points).shape[0]
                    pcd_single_view = pcd_single_view.random_down_sample(num_target_points / num_source_points)
                    #pcd_single_view = pcd_single_view.voxel_down_sample(0.02)

                    source_affordances, _ = getPredictedAffordancesInPointCloud(pcd_single_view)
                    source_affordances[0] = 0
                    target_affordances, _ = getPredictedAffordancesInPointCloud(pcd_affordance)

                    source_points = np.asanyarray(pcd_single_view.points)
                    source_colors = np.asanyarray(pcd_single_view.colors)
                    target_points = np.asanyarray(pcd_affordance.points)
                    target_colors = np.asanyarray(pcd_affordance.colors) * 255

                    if np.array_equal(source_affordances, target_affordances):

                        min_sum = None
                        min_sum_solution = None

                        initial_guess = self.get_random_quaternion()

                        target_aff = np.hstack((target_points, target_colors))
                        #source_aff = np.hstack((source_points, np.asanyarray(pcd_single_view.colors)))

                        #xopt = scipy.optimize.minimize(self.objective_function_with_affordances, x0 = initial_guess, method='Nelder-Mead', args=(target_points, target_colors, source_points, source_colors))
                        xopt = scipy.optimize.minimize(self.objective_function, x0 = initial_guess, method='Nelder-Mead', args=(target_points, source_points))

                        solution_mag = np.linalg.norm(xopt.x)
                        solution = xopt.x/solution_mag
                        quat = solution

                        rot_mat_sol = R.from_quat(quat).as_matrix()

                        pcd_score = o3d.geometry.PointCloud()
                        pcd_score.points = o3d.utility.Vector3dVector(source_points)
                        pcd_score.rotate(rot_mat_sol)

                        pcd_score_points = np.asanyarray(pcd_score.points)

                        neigh = NearestNeighbors(n_neighbors=1)
                        neigh.fit(target_points)
                        distances, indices = neigh.kneighbors(pcd_score_points, return_distance=True)

                        score = np.mean(distances) / num_source_points

                        centroid = pcd_affordance.get_center()
                        T = np.eye(4)
                        T[:3,:3] = rot_mat_sol
                        T[0, 3] = centroid[0]
                        T[1, 3] = centroid[1]
                        T[2, 3] = centroid[2]

                    else:
                        score = 9999
                        T = np.eye(4)


                    print(count, " / ", x_range*y_range*z_range, " best score so far: ", best_score)

                    if score < best_score:
                        pcd_single_view.transform(T)
                        transformation[:3, :3] = rot_mat
                        best_score = score
                        best_T = np.matmul(T, transformation)
                        best_T[0, 3] = pcd_single_view.get_center()[0]
                        best_T[1, 3] = pcd_single_view.get_center()[1]
                        best_T[2, 3] = pcd_single_view.get_center()[2]

                        o3d.visualization.draw_geometries([pcd_affordance, pcd_single_view])

                    if best_score < 0.02:
                        print("Best score below threshold")
                        x = x_range
                        y = y_range
                        z = z_range
                        break

                    count += 1

        te = time.time()
        print("Found transformation in: ", te -ts, " s")

        o3d.visualization.draw_geometries([pcd_affordance, source_pcd.transform(best_T)])
        return best_T, self.getGoalOrientation(predictions[0][0])

    def run(self, msg):

        print("received request...")

        camClient = CameraClient()
        pcd_geometry, pcd_color = camClient.unpackPCD(msg_geometry = msg.pcd_geometry,
                                                    msg_color = msg.pcd_color)

        pcd_affordance = o3d.geometry.PointCloud()
        pcd_affordance.points = o3d.utility.Vector3dVector(pcd_geometry)
        pcd_affordance.colors = o3d.utility.Vector3dVector(pcd_color)

        T = 0
        G = 0
        if self.method == 0:
            T, G = self.methodObservation(pcd_affordance)
            #T, G = self.methodObservationQuat(pcd_affordance)
        elif self.method == 1:
            T, G = self.methodRule(pcd_affordance)

        np.set_printoptions(suppress=True)
        print(T)
        print()
        print(G)

        msg = runOrientationSrvResponse()
        rotClient = OrientationClient()
        msg.current, msg.goal = rotClient.packOrientation(T, G)

        return msg

    def createNNClassifier(self):

        # bg, grasp, cut, scoop, contain, pound, support, w-grasp
        features = []
        features.append([0, 1, 0, 0, 0, 0, 1, 0]) # spatula, shovel
        features.append([0, 1, 1, 0, 0, 0, 0, 0]) # saw, knife, scissors, shears
        features.append([0, 1, 0, 0, 0, 1, 0, 0]) # hammer, mallet, tenderizers
        features.append([0, 1, 0, 1, 0, 0, 0, 0]) # scoop, spoon, trowel
        features.append([0, 1, 0, 0, 1, 0, 0, 0]) # ladle
        features.append([0, 0, 0, 0, 1, 0, 0, 1]) # bowl, cup
        features.append([0, 1, 0, 0, 1, 0, 0, 1]) # mug
        features.append([0, 1, 0, 0, 0, 0, 0, 0]) # bottle

        features = np.array(features)

        nn_classifier = NearestNeighbors(n_neighbors =  1).fit(features)

        return nn_classifier

    def getSourcePointCloud(self, prediction):

        path = self.pcd_paths[prediction]
        pcd = o3d.io.read_point_cloud(path, format='xyzrgb')

        return pcd

    def getGoalOrientation(self, prediction):

        rotation = R.from_quat(self.mean_quat[prediction])
        rotation_matrix = rotation.as_matrix()

        return rotation_matrix

    def computeFeatureVector(self, pcd):

        label_colors = {(0, 0, 255): 1, # grasp
        (0, 255, 0): 2, # cut
        (123, 255, 123): 3, # scoop
        (255, 0, 0): 4, # contain
        (255, 255, 0): 5, # pound
        (255, 255, 255): 6, # support
        (255, 0, 255): 7} # wrap-grasp

        feature_vector = np.zeros((1,8))

        pcd_colors = np.asanyarray(pcd.colors).astype(np.uint8)
        if np.max(pcd_colors) <= 1:
            pcd_colors = pcd_colors * 255

        for label_color in label_colors:
            idx = pcd_colors == label_color
            idx = np.sum(idx, axis = -1) == 3
            if True in idx:
                feature_vector[0, label_colors[label_color]] = 1

        return feature_vector


    def best_fit_transform(self, A, B):
        '''
        https://github.com/ClayFlannigan/icp
        Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
        Input:
          A: Nxm numpy array of corresponding points
          B: Nxm numpy array of corresponding points
        Returns:
          T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
          R: mxm rotation matrix
          t: mx1 translation vector
        '''

        assert A.shape == B.shape

        # get number of dimensions
        m = A.shape[1]

        # translate points to their centroids
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B

        # rotation matrix
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
           Vt[m-1,:] *= -1
           R = np.dot(Vt.T, U.T)

        # translation
        t = centroid_B.T - np.dot(R,centroid_A.T)

        # homogeneous transformation
        T = np.identity(m+1)
        T[:m, :m] = R
        T[:m, m] = t

        return T, R, t


    def nearest_neighbor(self, src, dst):
        '''
        https://github.com/ClayFlannigan/icp
        Find the nearest (Euclidean) neighbor in dst for each point in src
        Input:
            src: Nxm array of points
            dst: Nxm array of points
        Output:
            distances: Euclidean distances of the nearest neighbor
            indices: dst indices of the nearest neighbor
        '''

        assert src.shape == dst.shape

        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(src, return_distance=True)
        return distances.ravel(), indices.ravel()


    def icp(self, source_points, source_colors, target_points, target_colors, init_pose=None, max_iterations=100, tolerance=0.0001):
        '''
        https://github.com/ClayFlannigan/icp
        The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
        Input:
            A: Nxm numpy array of source mD points
            B: Nxm numpy array of destination mD point
            init_pose: (m+1)x(m+1) homogeneous transformation
            max_iterations: exit algorithm after max_iterations
            tolerance: convergence criteria
        Output:
            T: final homogeneous transformation that maps A on to B
            distances: Euclidean distances (errors) of the nearest neighbor
            i: number of iterations to converge
        '''

        if source_points.shape[0] > target_points.shape[0]:
            indices = t_indices = random.sample(range(0, source_points.shape[0]), target_points.shape[0])
            source_points = source_points[indices]
            source_colors = source_colors[indices]

        elif target_points.shape[0] > source_points.shape[0]:
            indices = t_indices = random.sample(range(0, target_points.shape[0]), source_points.shape[0])
            target_points = target_points[indices]
            target_colors = target_colors[indices]

        assert source_points.shape == target_points.shape
        A = source_points
        B = target_points

        # get number of dimensions
        m = A.shape[1]

        # make points homogeneous, copy them to maintain the originals
        src = np.ones((m+1,A.shape[0]))
        dst = np.ones((m+1,B.shape[0]))
        src[:m,:] = np.copy(A.T)
        dst[:m,:] = np.copy(B.T)

        # apply the initial pose estimation
        if init_pose is not None:
            src = np.dot(init_pose, src)

        prev_error = 0

        for i in range(max_iterations):
            # find the nearest neighbors between the current source and destination points
            src_aff = np.hstack((src[:m,:].T, source_colors))
            distances, indices = self.nearest_neighbor(np.hstack((src[:m,:].T, source_colors)),
                                                    np.hstack((dst[:m,:].T, target_colors)))

            #distances, indices = self.nearest_neighbor(src[:m,:].T, dst[:m,:].T)

            # compute the transformation between the current source and nearest destination points
            T,_,_ = self.best_fit_transform(src[:m,:].T, dst[:m,indices].T)



            # update the current source
            src = np.dot(T, src)

            # check error
            mean_error = np.mean(distances)
            if np.abs(prev_error - mean_error) < tolerance:
                break
            prev_error = mean_error

        # calculate final transformation
        T,_,_ = self.best_fit_transform(A, src[:m,:].T)

        return T, distances, i

    def icpWithNormals(self, source_points, source_colors, source_normals, target_points, target_colors, target_normals, init_pose=None, max_iterations=100, tolerance=0.0001):
        '''
        https://github.com/ClayFlannigan/icp
        The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
        Input:
            A: Nxm numpy array of source mD points
            B: Nxm numpy array of destination mD point
            init_pose: (m+1)x(m+1) homogeneous transformation
            max_iterations: exit algorithm after max_iterations
            tolerance: convergence criteria
        Output:
            T: final homogeneous transformation that maps A on to B
            distances: Euclidean distances (errors) of the nearest neighbor
            i: number of iterations to converge
        '''

        if source_points.shape[0] > target_points.shape[0]:
            indices = t_indices = random.sample(range(0, source_points.shape[0]), target_points.shape[0])
            source_points = source_points[indices]
            source_colors = source_colors[indices]
            source_normals = source_normals[indices]

        elif target_points.shape[0] > source_points.shape[0]:
            indices = t_indices = random.sample(range(0, target_points.shape[0]), source_points.shape[0])
            target_points = target_points[indices]
            target_colors = target_colors[indices]
            target_normals = target_normals[indices]

        assert source_points.shape == target_points.shape
        A = source_points
        B = target_points


        # get number of dimensions
        m = A.shape[1]

        # make points homogeneous, copy them to maintain the originals
        src = np.ones((m+1,A.shape[0]))
        dst = np.ones((m+1,B.shape[0]))
        src[:m,:] = np.copy(A.T)
        dst[:m,:] = np.copy(B.T)

        # apply the initial pose estimation
        if init_pose is not None:
            src = np.dot(init_pose, src)

        prev_error = 0

        for i in range(max_iterations):
            # find the nearest neighbors between the current source and destination points
            src_aff = np.hstack((src[:m,:].T, source_colors))
            #distances, indices = self.nearest_neighbor(np.hstack((src[:m,:].T, source_colors)),
            #                                        np.hstack((dst[:m,:].T, target_colors)))

            distances, indices = self.nearest_neighbor(np.hstack((source_normals, source_colors)),
                                                    np.hstack((source_normals, target_colors)))

            #distances, indices = self.nearest_neighbor(src[:m,:].T, dst[:m,:].T)

            # compute the transformation between the current source and nearest destination points
            T,_,_ = self.best_fit_transform(src[:m,:].T, dst[:m,indices].T)

            # update the current source
            src = np.dot(T, src)

            # check error
            mean_error = np.mean(distances)
            if np.abs(prev_error - mean_error) < tolerance:
                break
            prev_error = mean_error

        # calculate final transformation
        T,_,_ = self.best_fit_transform(A, src[:m,:].T)

        return T, distances, i

if __name__ == "__main__":
    global depth_image, rgb_image, capture


    rospy.init_node('orientation_service', anonymous=True)
    rate = rospy.Rate(10)

    orientation_predictor = OrientationServer()
    capture = False

    while True: # The offscreen renderer has to run on the main thread
        rate.sleep()
        if capture:
            depth_image = orientation_predictor.renderer.render_to_depth_image()
            rgb_image = orientation_predictor.renderer.render_to_image()

            capture = False

    while not rospy.is_shutdown():
        rospy.spin()
