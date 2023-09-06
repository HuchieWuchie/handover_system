#!/usr/bin/env python3
import sys
import numpy as np
import os, cv2
import argparse
import rospy


import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from lib.mask_rcnn import MaskRCNNPredictor, MaskAffordancePredictor, MaskRCNNHeads
import lib.mask_rcnn as mask_rcnn
import utils
import argparse
from PIL import Image
import numpy as np
import cv2
import time
from config import IITAFF
import os
from skimage.transform import resize
from affordanceService.client import AffordanceClient

from affordance_analyzer.srv import *
from std_msgs.msg import MultiArrayDimension, String


class AffordanceAnalyzer(object):
    """docstring for AffordanceAnalyzer."""

    def __init__(self):

        self.CONF_THRESHOLD = 0.7
        self.root_path = '/workspace/affordanceNet_sytn/'
        self.name = "affordancenet_synth"
        self.device = None

        self.serviceGet = rospy.Service('/affordance/result', getAffordanceSrv, self.getAffordance)
        self.serviceRun = rospy.Service('/affordance/run', runAffordanceSrv, self.analyzeAffordance)
        self.serviceStart = rospy.Service('/affordance/start', startAffordanceSrv, self.startAffordance)
        self.serviceStop = rospy.Service('/affordance/stop', stopAffordanceSrv, self.stopAffordance)
        self.serviceName = rospy.Service('/affordance/name', getNameSrv, self.getName)

        self.net = None

    def getName(self, msg):

        response = getNameSrvResponse()
        response.name = String(self.name)

        return response


    def run_net(self, x, CONF_THRESHOLD = 0.7):

        #ori_height, ori_width, _ = im.shape

        # Detect all object classes and regress object bounds
        #timer = Timer()
        #timer.tic()
        ts = time.time() * 1000

        predictions = self.net(x)[0]
        boxes, labels, scores, masks = predictions['boxes'], predictions['labels'], predictions['scores'], predictions['masks']
        print("Found: ")

        for label, score in zip(labels, scores):
            print(label, score)
        te = time.time() * 1000
        print("Prediction took: ", te - ts, " ms")
        try:
            idx = scores > CONF_THRESHOLD
            labels = labels.cpu().detach().numpy()


            boxes = boxes[idx].cpu().detach().numpy()
            labels = labels[idx.cpu().detach().numpy()]
            scores = scores[idx].cpu().detach().numpy()
            masks = masks[idx].cpu().detach().numpy()

            masks = masks * 255

        except:
            pass


        #timer.toc()
        #print ('Detection took {:.3f}s for '
        #       '{:d} object proposals').format(timer.total_time, rois_final.shape[0])

        try:
            return boxes, labels, masks, scores
        except:
            return 0

    def sendResults(self, bbox, objects, masks, scores):
        intToLabel = {0: 'class', 1: 'height', 2: 'width'}
        msg = getAffordanceSrvResponse()

        aff_client = AffordanceClient(connected = False)

        # constructing mask message
        """
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
        #msg.masks.data = masks
        """
        msg.masks = aff_client.packMasks(masks)

        # constructing bounding box message
        msg.bbox.data = bbox.flatten().astype(int).tolist()

        # constructing object detection class message
        msg.object.data = objects.flatten().tolist()

        # constructing the scores message
        msg.confidence.data = scores.flatten().tolist()

        return msg

    def analyzeAffordance(self, msg):

        img = np.frombuffer(msg.img.data, dtype=np.uint8).reshape(msg.img.height, msg.img.width, -1)
        width, height = img.shape[1], img.shape[0]
        ratio = width / height
        img = cv2.resize(img, (int(450 * ratio), 450), interpolation = cv2.INTER_AREA)
        x = [torchvision.transforms.ToTensor()(img).to(self.device)]
        self.CONF_THRESHOLD = msg.confidence_threshold.data

        print("Analyzing affordance with confidence threshold: ", self.CONF_THRESHOLD)
        try:
            bbox, objects, masks, scores = self.run_net(x, CONF_THRESHOLD=self.CONF_THRESHOLD)
            #bbox = bbox[:,1:]
            print(masks.shape, bbox.shape, objects.shape, scores.shape)
            try:
                m = np.zeros((11, height, width))
                for c, m_t in enumerate(masks[0]):
                    m[c] = resize(m_t, (height, width))
            except Exception as e:
                print(e)
                raise
            #m = resize(m, (width, height))
            #m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
            try:
                for c, mask in enumerate(masks):
                    if c > 0:
                        m_aff = np.zeros((11, height, width))
                        for c_a, aff_mask in enumerate(mask):
                            m_aff[c_a] = resize(aff_mask, (height, width))
                        m = np.vstack((m,m_aff))
            except Exception as e:
                print(e)
                raise
            masks = m

            for b_c, box in enumerate(bbox):
                box[0] = box[0] * (width / (450 * ratio))
                box[2] = box[2] * (width / (450 * ratio))
                box[1] = box[1] * (height / 450)
                box[3] = box[3] * (height / 450)
                bbox[b_c] = box


            self.bbox = bbox
            self.objects = objects
            self.masks = masks
            self.scores = scores
        except:
            bbox = np.zeros((1, 4))
            objects = np.zeros((1,1))
            masks = np.zeros((1, 11, 244, 244))
            scores = np.zeros(1)

        return runAffordanceSrvResponse()

    def getAffordance(self, msg):

        return self.sendResults(self.bbox, self.objects, self.masks, self.scores)

    def startAffordance(self, msg):
        #weights_path = os.path.dirname(os.path.realpath(__file__)) + "/14.pth"
        weights_path = os.path.dirname(os.path.realpath(__file__)) + "/weights.pth"

        GPU = msg.GPU.data
        device = torch.device('cpu')

        if GPU:
            device = torch.device('cuda')
        self.device = device
        print("Device is: ", self.device)

        model = utils.get_model_instance_segmentation(23, 11)
        #model = utils.get_model_instance_segmentation(18, 8)
        model.load_state_dict(torch.load(weights_path))
        model.to(device)
        model.eval()

        # load network
        self.net = model

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
