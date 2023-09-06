#!/usr/bin/env python
# modified from: https://github.com/nqanh/affordance-net/blob/master/tools/demo_img.py
import sys

import numpy as np
import os, cv2
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from affordance_analyzer.srv import *
from std_msgs.msg import MultiArrayDimension, String
import rospy

import caffe

sys.path.append('/affordance-net/tools')

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect2
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

class AffordanceAnalyzer(object):
    """docstring for AffordanceAnalyzer."""

    def __init__(self):

        self.CONF_THRESHOLD = 0.7
        self.good_range = 0.005
        self.cwd = os.getcwd()
        self.root_path = '/affordance-net/'
        self.img_folder = self.cwd + '/img'
        cfg.TEST.HAS_RPN = True
        self.name = "affordancenet"

        self.serviceGet = rospy.Service('/affordance/result', getAffordanceSrv, self.getAffordance)
        self.serviceRun = rospy.Service('/affordance/run', runAffordanceSrv, self.analyzeAffordance)
        self.serviceStart = rospy.Service('/affordance/start', startAffordanceSrv, self.startAffordance)
        self.serviceStop = rospy.Service('/affordance/stop', stopAffordanceSrv, self.stopAffordance)
        self.serviceName = rospy.Service('/affordance/name', getNameSrv, self.getName)

        self.net = None

        print 'AffordanceNet root folder: ', self.root_path

    def getName(self, msg):

        response = getNameSrvResponse()
        response.name = String(self.name)

        return response


    def run_affordance_net(self, im, CONF_THRESHOLD = 0.7):

        ori_height, ori_width, _ = im.shape

        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        if cfg.TEST.MASK_REG:
            rois_final, rois_class_score, rois_class_ind, masks, scores, boxes = im_detect2(self.net, im)
            print(rois_final, rois_class_score)
        else:
            1
        timer.toc()
        print ('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time, rois_final.shape[0])

        inds = np.where(rois_class_score[:, -1] >= CONF_THRESHOLD)[0]
        # get mask
        rois_final = rois_final[inds]
        rois_class_ind = rois_class_ind[inds]
        masks = masks[inds, :, :, :]
        masks = masks * 255
        num_boxes = rois_final.shape[0]
        im_width = im.shape[1]
        im_height = im.shape[0]

        masks_list = []
        for i in range(masks.shape[0]):
            bbox = rois_final[i]
            x1, y1, x2, y2 = int(bbox[1]), int(bbox[2]), int(bbox[3]), int(bbox[4])
            object_mask_list = []
            for j in range(masks.shape[1]):
                mask_resized = np.zeros((im_height, im_width))
                mask = masks[i, j, :, :]
                mask = cv2.resize(mask, (int(x2-x1), int(y2-y1)), interpolation=cv2.INTER_LINEAR)
                mask_resized[y1:y2, x1:x2] = mask
                object_mask_list.append(mask_resized)
            masks_list.append(object_mask_list)

        masks = np.array(masks_list)
        rois_class_score = np.array(rois_class_score).flatten()

        try:
            return rois_final, rois_class_ind, masks, rois_class_score
        except:
            return 0

    def sendResults(self, bbox, objects, masks, scores):
        intToLabel = {0: 'class', 1: 'height', 2: 'width'}
        msg = getAffordanceSrvResponse()

        # constructing mask message
        for i in range(3):
            dimMsg = MultiArrayDimension()
            dimMsg.label = intToLabel[i]
            stride = 1
            for j in range(3-i):
                stride = stride * masks.shape[i+j]
            dimMsg.stride = stride
            dimMsg.size = masks.shape[i]
            msg.masks.layout.dim.append(dimMsg)
        masks = masks.flatten().astype(int).tolist()
        msg.masks.data = masks

        # constructing bounding box message
        msg.bbox.data = bbox.flatten().astype(int).tolist()

        # constructing object detection class message
        msg.object.data = objects.flatten().tolist()

        # constructing the scores message
        msg.confidence.data = scores.flatten().tolist()

        return msg

    def analyzeAffordance(self, msg):

        img = np.frombuffer(msg.img.data, dtype=np.uint8).reshape(msg.img.height, msg.img.width, -1)

        self.CONF_THRESHOLD = msg.confidence_threshold.data

        print("Analyzing affordance with confidence threshold: ", self.CONF_THRESHOLD)
        try:
            bbox, objects, masks, scores = self.run_affordance_net(img, CONF_THRESHOLD=self.CONF_THRESHOLD)
            bbox = bbox[:,1:]
            m = masks[0]
            for i in range(len(masks)-1):
                m = np.vstack((m, masks[i+1]))
            masks = m

            print(bbox.shape, objects.shape, masks.shape)
            print(objects)
            self.bbox = bbox
            self.objects = objects
            self.masks = masks
            self.scores = scores
        except:
            bbox = np.zeros((1, 4))
            objects = np.zeros((1,1))
            masks = np.zeros((1, 10, 244, 244))
            scores = np.zeros(1)

        return runAffordanceSrvResponse()

    def getAffordance(self, msg):

        return self.sendResults(self.bbox, self.objects, self.masks, self.scores)

    def startAffordance(self, msg):
        prototxt = self.root_path + '/models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt'
        caffemodel = self.root_path + '/pretrained/AffordanceNet_200K.caffemodel'

        if not os.path.isfile(caffemodel):
            raise IOError(('{:s} not found.\n').format(caffemodel))

        GPU = msg.GPU.data
        if GPU:
            caffe.set_mode_gpu()
            caffe.set_device(0)
        else:
            caffe.set_mode_cpu()

        # load network
        self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        print '\n\nLoaded network {:s}'.format(caffemodel)

        msg = startAffordanceSrvResponse()
        return msg

    def stopAffordance(self, msg):
        del self.net
        self.net = None

        msg = stopAffordanceSrvResponse()
        return msg


if __name__ == '__main__':

    rospy.init_node('affordance_analyzer')
    affordanceServer = AffordanceAnalyzer()
    print("Affordance analyzer is ready.")

    rospy.spin()
