#!/usr/bin/env python3
import rospy
#from grasp_generator.srv import *
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import os
import math
from scipy.spatial.transform import Rotation as R

from affordanceService.client import AffordanceClient
from grasp_service.client import GraspingGeneratorClient
from rob9Utils.graspGroup import GraspGroup
from rob9Utils.visualize import visualizeGrasps6DOF, createGripper, visualizeGripper

from rob9Utils.affordancetools import getPointCloudAffordanceMask, getPredictedAffordances, getAffordanceContours, getObjectAffordancePointCloud
from rob9Utils.utils import erodeMask, keepLargestContour, convexHullFromContours, maskFromConvexHull, thresholdMaskBySize, removeOverlapMask


if __name__ == "__main__":
    print("Usage example of client server service")
    rospy.init_node('handover_orientation_client', anonymous=True)

    # Create grasp client

    grasp_client = GraspingGeneratorClient()

    # if you want to change the default grasping settings do so before starting
    # the network or running the grasping generator

    azimuth_step_size = 0.025
    azimuth_min = 0
    azimuth_max = 0.7

    polar_step_size = 0.05
    polar_min = 0.0
    polar_max = 0.2

    depth_step_size = 0.01 # m
    depth_min = 0
    depth_max = 0.03


    grasp_client.setSettings(azimuth_step_size, azimuth_min, azimuth_max,
                            polar_step_size, polar_min, polar_max,
                            depth_step_size, depth_min, depth_max)

    # Load sample data
    project_path = Path(os.path.realpath(__file__)).parent.parent.parent
    path = os.path.join(project_path, "sample_data")

    masks = np.load(os.path.join(path, "masks.npy")).astype(np.uint8)
    labels = np.load(os.path.join(path, "labels.npy"))
    bboxs = np.load(os.path.join(path, "bboxs.npy"))
    uv = np.load(os.path.join(path, "uv.npy"))
    pcd_colors = np.load(os.path.join(path, "cloudColor.npy"))
    pcd = o3d.io.read_point_cloud(os.path.join(path, "pcd.ply"))
    geometry = np.asanyarray(pcd.points)
    img = cv2.imread(os.path.join(path, "img.png"))

    # Select object 5, ladle or 0 for knife

    obj_inst = 5
    masks = masks[obj_inst]
    labels = labels[obj_inst]
    bbox = bboxs[obj_inst]

    # Post process affordance predictions and compute point cloud affordance mask

    affordances_in_object = getPredictedAffordances(masks = masks, bbox = bbox)
    print("predicted affordances", affordances_in_object)

    for aff in affordances_in_object:

        masks = erodeMask(affordance_id = aff, masks = masks,
                        kernel = np.ones((3,3)))
        contours = getAffordanceContours(bbox = bbox, affordance_id = aff,
                                        masks = masks)
        if len(contours) > 0:
            contours = keepLargestContour(contours)
            hulls = convexHullFromContours(contours)

            h, w = masks.shape[-2], masks.shape[-1]
            if bbox is not None:
                h = int(bbox[3] - bbox[1])
                w = int(bbox[2] - bbox[0])

            aff_mask = maskFromConvexHull(h, w, hulls = hulls)
            _, keep = thresholdMaskBySize(aff_mask, threshold = 0.05)
            if keep == False:
                aff_mask[:, :] = False

            if bbox is not None:
                masks[aff, bbox[1]:bbox[3], bbox[0]:bbox[2]] = aff_mask
            else:
                masks[aff, :, :] = aff_mask

    masks = removeOverlapMask(masks = masks)

    # transform point cloud into world coordinate frame
    # below is a transformation used during capture of sample data

    cam2WorldTransform = np.array([-(math.pi/2) - 1.0995574287564276, 0, -(math.pi)-(math.pi/4)+1.2220795422464295])
    rotCam2World = R.from_euler('xyz', cam2WorldTransform)
    rotMatCam2World = rotCam2World.as_matrix()

    points = np.dot(rotMatCam2World, np.asanyarray(pcd.points).T).T
    pcd.points = o3d.utility.Vector3dVector(points)

    # Compute a downsampled version of the point cloud for collision checking
    # downsampling speeds up computation
    pcd_downsample = pcd.voxel_down_sample(voxel_size=0.005)

    # Select affordance mask to compute grasps for
    observed_affordances = getPredictedAffordances(masks)

    affClient = AffordanceClient(connected = False)
    functional_labels = affClient.getFunctionalLabels()

    success, sampled_grasp_points = False, 0
    aff_id = 0
    for functional_label in functional_labels:
        if functional_label in observed_affordances:
            aff_id = functional_label

            success, sampled_grasp_points = getPointCloudAffordanceMask(affordance_id = functional_label,
                                            points = points, uvs = uv, masks = masks)

    if success:

        # run the algorithm
        grasp_group = grasp_client.run(sampled_grasp_points, pcd_downsample,
                                    "world", labels, aff_id, obj_inst)

        print(grasp_group.grasps[0])

        print("I got ", len(grasp_group.grasps), " grasps")

        grasp_group.thresholdByScore(0.3)
        grasp_group.sortByScore()
        best_grasp = grasp_group[0]

        rotMat = best_grasp.orientation.getRotationMatrix()
        translation = best_grasp.position.getVector()

        print(rotMat)
        print(translation)

        gripper = createGripper(opening = 0.08, translation = translation, rotation = rotMat)
        vis_gripper = visualizeGripper(gripper)




        print("I got ", len(grasp_group.grasps), " grasps after thresholding")

        vis_grasps = visualizeGrasps6DOF(pcd, grasp_group)
        o3d.visualization.draw_geometries([pcd, *vis_grasps, vis_gripper])

        #graspClient.visualizeGrasps(num_to_visualize = 0) # num_to_visualize = 0 visualizes all
        #cloud, rgb = cam.getPointCloudStatic()
