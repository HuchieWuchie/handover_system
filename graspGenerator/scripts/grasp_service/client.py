#!/usr/bin/env python3
import rospy
from grasp_generator.srv import *
from rob9.msg import GraspMsg, GraspGroupMsg
from rob9.srv import *
from std_msgs.msg import Float32, Int32, String
import numpy as np
import cv2
import open3d as o3d
import copy
from cameraService.cameraClient import CameraClient
from rob9Utils.graspGroup import GraspGroup, Grasp

class GraspingGeneratorClient(object):
    """docstring for GraspingGeneratorClient."""

    def __init__(self):

        self.azimuth_step_size = 0.025
        self.azimuth_min = 0
        self.azimuth_max = 0.7

        self.polar_step_size = 0.05
        self.polar_min = 0.0
        self.polar_max = 0.2

        self.depth_step_size = 0.01 # m
        self.depth_min = 0
        self.depth_max = 0.03

    def run(self, sampled_grasp_points, pcd_environment, frame_id, tool_id,
            affordance_id, object_instance):
        """ Input:
            sampled_grasp_points    - np.array(), shape(N, 3) x, y, z
            pcd_environment         - open3d.geometry.PointCloud()

            Output:
            grasps                  - rob9.GraspGroup()
        """

        print("Waiting for grasp service")
        rospy.wait_for_service("/iiwa/grasp_generator/result")
        print("Grasp service is up, generating grasps...")
        graspGeneratorService = rospy.ServiceProxy("/iiwa/grasp_generator/result", runGraspingSrv)

        cam_client = CameraClient()

        pcd_points = np.asanyarray(pcd_environment.points)
        pcd_msg, _ = cam_client.packPCD(pcd_points, None)
        grasp_points_msg, _ = cam_client.packPCD(sampled_grasp_points, None)

        response = graspGeneratorService(grasp_points_msg, pcd_msg, String(frame_id),
                                        Int32(tool_id), Int32(affordance_id),
                                        Int32(object_instance))

        grasps = GraspGroup().fromGraspGroupMsg(response)

        return grasps

    def packGrasps(self, poses, scores, frame_id, tool_id,
                                        affordance_id, obj_inst):

        grasps = []
        for pose, score in zip(poses, scores):

            grasp = Grasp(frame_id = frame_id)
            grasp.position.set(x = pose[0], y = pose[1], z = pose[2])
            grasp.orientation.setQuaternion(pose[3:])
            grasp.score = score
            grasp.tool_id = tool_id
            grasp.affordance_id = affordance_id
            grasp.setObjectInstance(obj_inst)

            grasp_msg = grasp.toGraspMsg()
            grasps.append(grasp_msg)

        return grasps

    def unpackGrasps(self, msg):
        todo = True

    def setSettings(self, azimuth_step_size, azimuth_min, azimuth_max,
                            polar_step_size, polar_min, polar_max,
                            depth_step_size, depth_min, depth_max):

        self.azimuth_step_size = azimuth_step_size
        self.azimuth_min = azimuth_min
        self.azimuth_max = azimuth_max

        self.polar_step_size = polar_step_size
        self.polar_min = polar_min
        self.polar_max = polar_max

        self.depth_step_size = depth_step_size # m
        self.depth_min = depth_min
        self.depth_max = depth_max

        rospy.wait_for_service("/iiwa/grasp_generator/set_settings")
        graspGeneratorService = rospy.ServiceProxy("/iiwa/grasp_generator/set_settings", setSettingsGraspingSrv)

        response = graspGeneratorService(Float32(azimuth_step_size), Int32(azimuth_min), Float32(azimuth_max),
                                        Float32(polar_step_size), Int32(polar_min), Float32(polar_max),
                                        Float32(depth_step_size), Int32(depth_min), Float32(depth_max))
