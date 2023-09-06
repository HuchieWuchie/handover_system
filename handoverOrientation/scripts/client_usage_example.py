#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
import open3d as o3d
import os
import time
from pathlib import Path

from cameraService.cameraClient import CameraClient
from affordanceService.client import AffordanceClient
from orientationService.client import OrientationClient

from rob9Utils.visualize import visualizeMasksInRGB, visualizeFrameMesh
from rob9Utils.affordancetools import getPredictedAffordances, getAffordanceContours, getObjectAffordancePointCloud
from rob9Utils.utils import erodeMask, keepLargestContour, convexHullFromContours, maskFromConvexHull, thresholdMaskBySize, removeOverlapMask



if __name__ == "__main__":
    print("Usage example of client server service")
    rospy.init_node('handover_orientation_client', anonymous=True)

    rotClient = OrientationClient()

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
    """
    cam = CameraClient()

    cam.captureNewScene()
    geometry, pcd_colors = cam.getPointCloudStatic()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(geometry)
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

    uv = cam.getUvStatic()
    img = cam.getRGB()

    affClient = AffordanceClient()

    affClient.start(GPU=True)
    _ = affClient.run(img, CONF_THRESHOLD = 0.8)

    masks, labels, scores, bboxs = affClient.getAffordanceResult()
    masks = affClient.processMasks(masks, conf_threshold = 0, erode_kernel=(1,1))
    """

    # Select object 5, ladle or 0 for knife

    masks = masks[0]
    labels = labels[0]
    bbox = bboxs[0]

    # Visualize masks
    cv2.imshow("Masks", visualizeMasksInRGB(img, masks))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Perform post processing of affordance segmentation

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

    pcd_affordance = getObjectAffordancePointCloud(pcd, masks, uvs = uv)

    # Compute orientation
    print("computing orientation")

    ts = time.time()
    rotClient.setSettings(0)
    orientation, translation, goal = rotClient.getOrientation(pcd_affordance)
    te = time.time()

    t_2_f = te - ts
    print("Computed current transformation and goal orientation in: ", round(t_2_f, 2), " seconds")

    print(orientation)
    print(translation)
    print(goal)

    # Visualize computed orientation
    rotated_coordinate_frame = visualizeFrameMesh(translation, orientation)
    #rotated_coordinate_frame_to_goal = visualizeFrameMesh(translation, goal)

    o3d.visualization.draw_geometries([pcd, rotated_coordinate_frame])
