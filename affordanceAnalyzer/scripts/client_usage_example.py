#!/usr/bin/env python3
import rospy
from affordance_analyzer.srv import getAffordanceSrv, getAffordanceSrvResponse
import numpy as np
import cv2
from cameraService.cameraClient import CameraClient
from affordanceService.client import AffordanceClient
from rob9Utils.visualize import visualizeMasksInRGB, visualizeBBoxInRGB
from rob9Utils.affordancetools import getPredictedAffordances, getAffordanceContours
from rob9Utils.utils import erodeMask, keepLargestContour, convexHullFromContours, maskFromConvexHull, thresholdMaskBySize, removeOverlapMask


import open3d as o3d
import copy

if __name__ == "__main__":
    print("Usage example of client server service")
    rospy.init_node('affordance_analyzer_client', anonymous=True)

    affClient = AffordanceClient()
    cam = CameraClient()
    cam.captureNewScene()
    cloud, cloudColor = cam.getPointCloudStatic()
    cloud_uv = cam.getUvStatic()
    img = cam.getRGB()

    cv2.imshow("name", img)
    cv2.waitKey(0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    pcd.colors = o3d.utility.Vector3dVector(cloudColor)
    affClient.start(GPU=True) # GPU acceleratio True or False
    print(affClient.name)
    print("Telling affordanceNET to analyze image from realsense and return predictions.")
    success = affClient.run(img, CONF_THRESHOLD = 0.7)
    masks, labels, scores, bboxs = affClient.getAffordanceResult()
    print("Got result")

    print(scores)

    masks = affClient.processMasks(masks, conf_threshold = 0, erode_kernel=(1,1))
    print("Here", masks.shape)

    # Post process affordance predictions and compute point cloud affordance mask

    for i in range(masks.shape[0]):

        obj_inst_masks = masks[i]
        obj_inst_bbox = bboxs[i]
        affordances_in_object = getPredictedAffordances(masks = masks[i], bbox = bboxs[i])
        print("predicted affordances", affordances_in_object)

        for aff in affordances_in_object:

            m_vis = np.zeros(obj_inst_masks.shape)
            #mask_eroded = erodeMask(affordance_id = aff, masks = obj_inst_masks,
            #                kernel = np.ones((3,3)))
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

            #obj_inst_masks = removeOverlapMask(masks = obj_inst_masks)
        print(masks[i].shape, obj_inst_masks.shape)
        masks[i] = obj_inst_masks

    img_vis = visualizeBBoxInRGB(visualizeMasksInRGB(img, obj_inst_masks), labels, bboxs, scores)
    cv2.imwrite("aff_detection.png", img_vis)
    cv2.imwrite("aff_input.png", img)
    cv2.imshow("masks", img_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if True:
        np.save("masks.npy", masks.astype(np.bool))
        np.save("labels.npy", labels)
        np.save("bboxs.npy", bboxs)
        np.save("scores.npy", scores)
        np.save("cloudColor.npy", cloudColor)
        np.save("uv.npy", cloud_uv)
        cv2.imwrite("img.png", img)
        o3d.io.write_point_cloud("pcd.ply", pcd)


    clouds_m = []

    cv2.imwrite("img.png", img)

    print("Found the following objects")
    for i in range(len(affClient.objects)):
        print(affClient.OBJ_CLASSES[affClient.objects[i]])
