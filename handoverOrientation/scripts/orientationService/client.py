#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray, Int32MultiArray, Int32
import numpy as np
from cameraService.cameraClient import CameraClient
from affordanceService.client import AffordanceClient
from orientation_service.srv import runOrientationSrv, runOrientationSrvResponse
from orientation_service.srv import setSettingsOrientationSrv, setSettingsOrientationSrvResponse

class OrientationClient(object):
    """docstring for orientationClient."""

    def __init__(self):

        self.method = 0 # learned from observation

    def getOrientation(self, pcd_affordance):

        print("Waiting for orientation service...")
        rospy.wait_for_service("/computation/handover_orientation/get")
        print("Orientation service is up...")
        orientationService = rospy.ServiceProxy("/computation/handover_orientation/get", runOrientationSrv)
        print("Connection to orientation service established!")

        camClient = CameraClient()
        affClient = AffordanceClient(connected = False)

        pcd_points = np.asanyarray(pcd_affordance.points)
        pcd_colors = np.asanyarray(pcd_affordance.colors)

        if np.max(pcd_colors) <= 1:
            pcd_colors = pcd_colors * 255

        pcd_geometry_msg, pcd_color_msg = camClient.packPCD(pcd_points, pcd_colors)

        print("Message constructed")

        response = orientationService(pcd_geometry_msg, pcd_color_msg)

        current_orientation, current_translation, goal_orientation = self.unpackOrientation(response.current, response.goal)

        del response

        return current_orientation, current_translation, goal_orientation

    def setSettings(self, method):

        if method == 0 or method == 1:
            self.method = method

            rospy.wait_for_service("/computation/handover_orientation/set_settings")
            settingsService = rospy.ServiceProxy("/computation/handover_orientation/set_settings", setSettingsOrientationSrv)

            _ = settingsService(Int32(method))

        else:
            print("Invalid method")

    def packOrientation(self, current_transformation, goal_orientation):

        msg_current = Float32MultiArray()
        current_transformation = current_transformation.flatten().tolist()
        msg_current.data = current_transformation

        msg_goal = Float32MultiArray()
        goal_orientation = goal_orientation.flatten().tolist()
        msg_goal.data = goal_orientation

        return msg_current, msg_goal

    def unpackOrientation(self, msg_current, msg_goal):
        current_transformation = np.asarray(msg_current.data).reshape((4,4))
        orientation = current_transformation[:3,:3]
        translation = current_transformation[:3,3]

        goal = np.asarray(msg_goal.data).reshape((3,3))
        return orientation, translation, goal
