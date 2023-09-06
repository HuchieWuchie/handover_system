#!/usr/bin/env python3

import sys
import time
import copy
import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray, Int32MultiArray, MultiArrayDimension, String, UInt8MultiArray

from cameraService.cameraClient import CameraClient
from affordance_analyzer.srv import *

def get_optimal_font_scale(text, width):
    """https://stackoverflow.com/questions/52846474/how-to-resize-text-for-cv2-puttext-according-to-the-image-size-in-opencv-python"""
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            if scale/10 > 1.0:
                return 1
            return scale/10
    return 1

class AffordanceClient(object):
    """docstring for AffordanceClient."""

    def __init__(self, connected = True):

        self.no_objects = 0
        self.masks = None
        self.bbox = None
        self.objects = None
        self.scores = None
        self.GPU = False

        self.name = "affordancenet_synth"
        if connected:
            self.name = self.getName()

        if self.name == "affordancenet":
            self.noObjClass = 11
            self.noLabelClass = 10
            self.OBJ_CLASSES = ('__background__', 'bowl', 'tvm', 'pan', 'hammer', 'knife',
                                    'cup', 'drill', 'racket', 'spatula', 'bottle')
            self.OBJ_NAME_TO_ID = {'__background__': 0, 'bowl': 1, 'tvm': 2, 'pan': 3,
                                'hammer': 4, 'knife': 5, 'cup': 6, 'drill': 7,
                                'racket': 8, 'spatula': 9, 'bottle': 10}
            self.labelsToNames = {'__background__': 0, 'contain': 1, 'cut': 2, 'display': 3, 'engine': 4, 'grasp': 5, 'hit': 6, 'pound': 7, 'support': 8, 'w-grasp': 9}
            self.graspLabels = [5, 9]
            self.functionalLabels = [1, 2, 3, 4, 6, 7, 8]

        elif self.name == "affordancenet_context":
            self.noObjClass = 2
            self.noLabelClass = 8
            self.OBJ_CLASSES = ('__background__', 'objectness')
            self.OBJ_NAME_TO_ID = {'__background__': 0, 'objectness': 1}
            self.labelsToNames = {'__background__': 0, 'grasp': 1, 'cut': 2, 'scoop': 3, 'contain': 4, 'pound': 5, 'support': 6, 'w-grasp': 7}
            self.graspLabels = [1, 7]
            self.functionalLabels = [2, 3, 4, 5, 6]

        elif self.name == "affordancenet_synth":
            self.noObjClass = 23
            self.noLabelClass = 11
            self.OBJ_CLASSES = ('__background__', 'knife', 'saw', 'scissors', 'shears', 'scoop',
                                        'spoon', 'trowel', 'bowl', 'cup', 'ladle',
                                        'mug', 'pot', 'shovel', 'turner', 'hammer',
                                        'mallet', 'tenderizer', 'bottle', 'drill', 'monitor', 'pan', 'racket')
            self.OBJ_NAME_TO_ID = {'__background__': 0, 'knife': 1, 'saw': 2, 'scissors': 3, 'shears': 4, 'scoop': 5,
                                'spoon': 6, 'trowel': 7, 'bowl': 8, 'cup': 9, 'ladle': 10,
                                'mug': 11, 'pot': 12, 'shovel': 13, 'turner': 14, 'hammer': 15,
                                'mallet': 16, 'tenderizer': 17, 'bottle': 18, 'drill': 19, 'tvm': 20, 'pan': 21, 'racket': 22}
            self.NamesToLabels = {'__background__': 0, 'grasp': 1, 'cut': 2, 'scoop': 3, 'contain': 4, 'pound': 5, 'support': 6, 'wrap-grasp': 7, 'display': 8, 'engine': 9, 'hit': 10}
            self.labelsToNames = {0: '__background__', 1: 'grasp', 2: 'cut', 3: 'scoop', 4: 'contain', 5: 'pound', 6: 'support', 7: 'wrap-grasp', 8: 'display', 9: 'engine', 10: 'hit'}
            self.graspLabels = [1]
            self.functionalLabels = [2, 3, 4, 5, 6, 7, 8, 9]

    def getName(self):

        rospy.wait_for_service("/affordance/name")
        nameService = rospy.ServiceProxy("/affordance/name", getNameSrv)
        msg = getNameSrv()
        msg.data = True
        response = nameService(msg)

        return response.name.data

    def getFunctionalLabels(self):
        return self.functionalLabels


    def start(self, GPU=False):
        self.GPU = GPU

        rospy.wait_for_service("/affordance/start")
        startAffordanceNetService = rospy.ServiceProxy("/affordance/start", startAffordanceSrv)
        msg = startAffordanceSrv()
        msg.data = GPU
        response = startAffordanceNetService(msg)

        return response.status.data


    def stop(self):
        rospy.wait_for_service("/affordance/stop")
        stopAffordanceNetService = rospy.ServiceProxy("/affordance/stop", stopAffordanceSrv)
        msg = stopAffordanceSrv()
        msg.data = True
        response = startAffordanceNetService(msg)

        return response.status.data

    def run(self, img, CONF_THRESHOLD = 0.7):
        rospy.wait_for_service("/affordance/run")
        runAffordanceNetService = rospy.ServiceProxy("/affordance/run", runAffordanceSrv)

        imgMsg = Image()
        imgMsg.height = img.shape[0]
        imgMsg.width = img.shape[1]
        imgMsg.data = img.flatten().tolist()

        response = runAffordanceNetService(imgMsg, Float32(CONF_THRESHOLD))

        return response.success.data

    def getAffordanceResult(self):

        s1 = time.time()
        rospy.wait_for_service("/affordance/result")
        affordanceNetService = rospy.ServiceProxy("/affordance/result", getAffordanceSrv)
        #print("Waiting for affordance service took: ", (time.time() - s1) * 1000 )

        s2 = time.time()
        msg = getAffordanceSrv()
        msg.data = True
        response = affordanceNetService(msg)

        self.masks = self.unpackMasks(response.masks)
        self.no_objects = self.masks.shape[0]
        self.bbox = self.unpackBBox(response.bbox)
        self.objects = self.unpackObjects(response.object)
        self.scores = self.unpackScores(response.confidence)

        return self.masks, self.objects, self.scores, self.bbox

    def visualizeBBox(self, im, labels, bboxs, scores):

        if bboxs is None:
            print("No bounding boxes to visualize")
            return 0
        if bboxs.shape[0] < 0:
            print("No bounding boxes to visualize")
            return 0

        img = im.copy()

        for box, label, score in zip(bboxs, labels, scores):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            ps = (box[0], box[1])
            pe = (box[2], box[3])
            color = (0, 0, 255)
            thickness = 2
            img = cv2.rectangle(img, ps, pe, color, thickness)

            text = str(self.OBJ_CLASSES[label]) + " " + str(round(score, 2))
            width = int(box[2] - box[0])
            fontscale = get_optimal_font_scale(text, width)
            y_rb = int(box[3] + 40)
            img[int(box[3]):y_rb, int(box[0])-1:int(box[2])+2] = (0, 0, 255)
            img = cv2.putText(img, text, (box[0], int(y_rb-10)), cv2.FONT_HERSHEY_SIMPLEX, fontscale, (255, 255, 255), 2, 2)

        return img

    def processMasks(self, masks, conf_threshold = 50, erode_kernel=(21,21)):
        if masks is None:
            print("No masks available to process")
            return 0
        if masks.shape[0] < 0:
            print("No masks available to process")
            return 0

        kernel = np.ones(erode_kernel, np.uint8)

        #full_mask = np.zeros((self.masks.shape[1], self.masks.shape[2], self.masks.shape[3])).astype(np.uint8)
        m = np.zeros(masks.shape)
        for i in range(masks.shape[0]):
            mask_arg = np.argmax(masks[i], axis = 0)
            color_idxs = np.unique(mask_arg)
            for color_idx in color_idxs:
                if color_idx != 0:
                    m[i, color_idx, mask_arg == color_idx] = 1
        for i in range(m.shape[0]):
            for j in range(m[i].shape[0]):
                m[i,j] = cv2.erode(m[i, j], kernel)
        return m

    def unpackMasks(self, msg):

        no_objects = int(msg.layout.dim[0].size / self.noLabelClass)
        masks = np.asarray(msg.data).reshape((no_objects, int(msg.layout.dim[0].size / no_objects), msg.layout.dim[1].size, msg.layout.dim[2].size)) #* 255
        masks = masks.astype(np.uint8)

        return masks

    def packMasks(self, masks):

        if len(masks.shape) <= 3:
            masks = np.reshape(masks, (-1, masks.shape[0], masks.shape[1], masks.shape[2]))

        msg = Int32MultiArray()
        intToLabel = {0: 'class', 1: 'height', 2: 'width'}

        for i in range(3):
            dimMsg = MultiArrayDimension()
            dimMsg.label = intToLabel[i]
            stride = 1
            for j in range(3-i):
                stride = stride * masks.shape[i+j + 1]
            dimMsg.stride = stride
            dimMsg.size = masks.shape[i + 1]
            msg.layout.dim.append(dimMsg)
        masks = masks.flatten().astype(int).tolist()
        msg.data = masks

        return msg

    def unpackBBox(self, msg):
        return np.asarray(msg.data).reshape((-1,4))

    def packBbox(self, bbox):

        msg = Int32MultiArray()
        msg.data = bbox.flatten().astype(int).tolist()

        return msg

    def unpackObjects(self, msg):
        return np.asarray(msg.data)

    def packObjects(self, objects):

        msg = Int32MultiArray()
        msg.data = objects.flatten().tolist()

        return msg

    def unpackScores(self, msg):
        return np.asarray(msg.data)

    def packScores(self, scores):

        msg = Float32MultiArray()
        msg.data = scores.flatten().tolist()
