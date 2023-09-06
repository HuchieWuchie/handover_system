#!/usr/bin/env python3
import rospy
from std_msgs.msg import Bool
import numpy as np
from location_service.srv import requestReceiverPose, requestReceiverPoseResponse

class LocationClient(object):
    """docstring for locationClient."""

    def __init__(self):
        pass
        #rospy.init_node('location_service_client', anonymous=True)

    def getLocation(self):

        rospy.wait_for_service('/iiwa/requestReceiverPose')
        get_receiver_pose = rospy.ServiceProxy("/iiwa/requestReceiverPose", requestReceiverPose)

        req = Bool()
        req.data = True
        resp = get_receiver_pose(req)

        location = self.unpackLocation(resp.receiver)

        return location

    def unpackLocation(self, msg):
        return np.array([msg.x, msg.y, msg.z]).reshape((3,1))
