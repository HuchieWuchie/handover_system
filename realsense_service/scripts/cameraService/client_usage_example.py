#!/usr/bin/env python3

"""
Import sys modular, sys.argv The function of is to pass parameters from the outside to the inside of the program. sys.argv(number)，number=0 Is the name of the script
"""
import sys
import rospy
from realsense_service.srv import *
import numpy as np
import cv2
import open3d as o3d
from cv_bridge import CvBridge
from cameraClient import CameraClient
import rob9Utils.iiwa
import rob9Utils.moveit as moveit

import datetime

if __name__ == "__main__":


    print("Starting")
    cam = CameraClient(type = "realsenseD435")
    counter = 0
    rospy.init_node("realsense_client_usage_example", anonymous=True)

    set_ee = True
    if not rob9Utils.iiwa.setEndpointFrame():
        set_ee = False
    print("STATUS end point frame was changed: ", set_ee)

    set_PTP_speed_limit = True
    if not rob9Utils.iiwa.setPTPJointSpeedLimits(0.1, 0.1):
        set_PTP_speed_limit = False
    print("STATUS PTP joint speed limits was changed: ", set_PTP_speed_limit)

    set_PTP_cart_speed_limit = True
    if not rob9Utils.iiwa.setPTPCartesianSpeedLimits(0.1, 0.1, 0.1, 0.1, 0.1, 0.1):
        set_PTP_cart_speed_limit = False
    print("STATUS PTP cartesian speed limits was changed: ", set_PTP_cart_speed_limit)


    rob9Utils.iiwa.execute_ptp(moveit.getJointPositionAtNamed("ready").joint_position.data)

    while(counter<1000):
        print("Capture new scene " + str(counter))
        counter += 1;

        if (counter % 2 == 0):
            rob9Utils.iiwa.execute_ptp(moveit.getJointPositionAtNamed("ready").joint_position.data)
        else:
            result = rob9Utils.iiwa.execute_ptp(moveit.getJointPositionAtNamed("camera_ready_1").joint_position.data)


        cam.captureNewScene()

        cam.getRGB()
        #cv2.imshow("rgb image", cam.rgb)
        #cv2.waitKey(0)

        # SOMETHING WRONG in cam.getDepth() -> ... dtype=np.float16 ...
        cam.getDepth()
        #print(cam.depth)
        #print(np.unique(cam.depth))
        #cv2.imshow("depth image", (cam.depth*255).astype(np.uint8))
        #cv2.waitKey(0)

        cam.getUvStatic()
        #print(cam.uv.shape)
        #print(cam.uv[0:10000])

        cloud, rgb = cam.getPointCloudStatic()
        #pcd = o3d.geometry.PointCloud()
        #pcd.points = o3d.utility.Vector3dVector(cloud)
        #pcd.colors = o3d.utility.Vector3dVector(rgb)
        #o3d.visualization.draw_geometries([pcd])


        sleep = np.random.randint(180, 300)
        time_now = rospy.Time.now()
        dt = datetime.datetime.utcfromtimestamp(time_now.to_sec())
        print("Time now ", dt)
        print("Sleeping for ", sleep)
        print("===========================")
        rospy.sleep(sleep)
