#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from geometry_msgs.msg import Pose, Pose
from std_msgs.msg import String, Float32, Int32, Header
from rob9.msg import GraspMsg, GraspGroupMsg
from rob9.srv import graspGroupSrv, graspGroupSrvResponse
from rob9Utils.grasp import Grasp

class GraspGroup(object):
    """docstring for GraspGroup."""

    def __init__(self, grasps = []):

        if type(grasps) is not list:
            print("Only accepts grasps as list")
            return None
        self.grasps = grasps

    def __getitem__(self, index):
        return self.grasps[index]

    def __setitem__(self, index, item):
        self.grasps[index] = item

    def __len__(self):
        return len(self.grasps)

    def add(self, grasp):
        self.grasps.append(grasp)

    def combine(self, other):

        for grasp in other.grasps:
            self.add(grasp)

        return self

    def fromPath(self, msg):
        self.__init__()
        self.grasps = []
        for poseMsg in msg.poses:
            self.add(Grasp().fromPoseStampedMsg(poseMsg))

        return self

    def fromGraspGroupMsg(self, msg):
        self.__init__()
        self.grasps = []
        for graspMsg in msg.grasps:
            self.add(Grasp().fromGraspMsg(graspMsg))

        return self

    def fromGraspGroupSrv(self, msg):

        self.__init__()
        self.grasps = []
        for graspMsg in msg.grasps:
            self.add(Grasp().fromGraspMsg(graspMsg))

        return self

    def getgraspsByAffordanceLabel(self, label):

        grasps = []
        for grasp in self.grasps:
            if grasp.affordance_id == label:
                grasps.append(grasp)

        return grasps

    def getGraspsByFrame(self, frame):

        grasps = []
        for grasp in self.grasps:
            if grasp.frame_id == frame:
                grasps.append(grasp)

        return grasps

    def getGraspsByInstance(self, instance):

        grasps = []
        for grasp in self.grasps:
            if grasp.object_instance == instance:
                grasps.append(grasp)

        return grasps

    def getgraspsByTool(self, id):

        grasps = []
        for grasp in self.grasps:
            if grasp.tool_id == id:
                grasps.append(grasp)

        return grasps

    def setAffordanceID(self, id):

        for grasp in self.grasps:
            grasp.affordance_id = id

    def setFrameId(self, id):

        for grasp in self.grasps:
            grasp.frame_id = str(id)

    def setObjectInstance(self, instance):

        for grasp in self.grasps:
            grasp.object_instance = instance

    def setToolId(self, id):

        for grasp in self.grasps:
            grasp.tool_id = id

    def sortByScore(self):

        scores = []
        for grasp in self.grasps:
            scores.append(grasp.score)

        idx = sorted(range(len(scores)), key=lambda k: scores[k])
        idx.reverse()
        grasps = [self.grasps[i] for i in idx]
        self.grasps = grasps

    def thresholdByScore(self, thresh):

        thresholdGrasps = []
        for grasp in self.grasps:
            if grasp.score >= thresh:
                thresholdGrasps.append(grasp)
        self.grasps = thresholdGrasps

    def toGraspGroupMsg(self):
        """ returns a graspGroup message """

        #msg = GraspGroupMsg() # outcommented Nov 29
        graspList = []

        for grasp in self.grasps:
            graspList.append(grasp.toGraspMsg())

        #msg.grasps = graspList
        return graspList

    def toGraspGroupSrv(self):
        """ returns a graspGroup message """

        msg = graspGroupSrvResponse()
        graspList = []

        for grasp in self.grasps:
            graspList.append(grasp.toGraspMsg())

        msg.grasps = graspList
        return msg

    def transformToFrame(self, tf_buffer, frame_dest):

        grasps = []
        for grasp in self.grasps:
            grasp.transformToFrame(tf_buffer, frame_dest)
