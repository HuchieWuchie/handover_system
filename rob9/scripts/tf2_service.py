#!/usr/bin/env python
import rospy
import geometry_msgs
from geometry_msgs.msg import PoseStamped, Pose, TransformStamped
from nav_msgs.msg import Path
import tf2_ros
import tf2_geometry_msgs

from rob9.srv import tf2TransformPoseStampedSrv, tf2TransformPoseStampedSrvResponse
from rob9.srv import tf2TransformPathSrv, tf2TransformPathSrvResponse
from rob9.srv import tf2GetTransformSrv, tf2GetTransformSrvResponse
from rob9.srv import tf2VisualizeTransformSrv, tf2VisualizeTransformSrvResponse
#from rob9.srv import tf2getQuatSrv, tf2getQuatSrvResponse


def visualizeTransform(msg):

    name = msg.name
    transform = msg.transform

    broadcaster = tf2_ros.StaticTransformBroadcaster()
    rospy.sleep(1.0)
    transform_stamped = TransformStamped()

    transform_stamped.header.stamp = rospy.Time.now()
    transform_stamped.header.frame_id = "world"
    transform_stamped.child_frame_id = name.data
    transform_stamped.transform = transform

    broadcaster.sendTransform(transform_stamped)

    return tf2VisualizeTransformSrvResponse()





def transformPathToFrame(req):

    newFrame = req.new_frame.data

    transformed_poses = []
    for pose in req.poses:
        pose.header.stamp = rospy.Time.now()
        transformed_pose_msg = geometry_msgs.msg.PoseStamped()
        tf_buffer.lookup_transform(pose.header.frame_id, newFrame, rospy.Time.now(), rospy.Duration(1))
        transformed_poses.append(tf_buffer.transform(pose, newFrame))

    path = Path()
    path.poses = transformed_poses

    return transformed_pose_msg


def getTransform(req):

    source_frame = req.source_frame.data
    target_frame = req.target_frame.data

    try:
        transformation_msg = tf_buffer.lookup_transform(target_frame, source_frame,
                                                    rospy.Time(0), rospy.Duration(0.1))
    except (tf2_ros.LookUpException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolatioException):
        rospy.logerr('Unable to find transformation')

    response = tf2GetTransformSrvResponse()
    response.transform = transformation_msg.transform

    return response


def transformToFrame(req):

    pose = req.pose
    newFrame = req.new_frame.data

    pose.header.stamp = rospy.Time.now()
    transformed_pose_msg = geometry_msgs.msg.PoseStamped()
    tf_buffer.lookup_transform(pose.header.frame_id, newFrame, rospy.Time.now(), rospy.Duration(1))
    transformed_pose_msg = tf_buffer.transform(pose, newFrame)
    return transformed_pose_msg


if __name__ == '__main__':

    baseServiceName = "/tf2/"

    rospy.init_node('tf2_service', anonymous=True)
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    transformPoseStampedService = rospy.Service(baseServiceName + "transformPoseStamped", tf2TransformPoseStampedSrv, transformToFrame)
    transformPathService = rospy.Service(baseServiceName + "transformPath", tf2TransformPathSrv, transformToFrame)
    getTransformService = rospy.Service(baseServiceName + "get_transform", tf2GetTransformSrv, getTransform)
    getTransformService = rospy.Service(baseServiceName + "visualize_transform", tf2VisualizeTransformSrv, visualizeTransform)

    rospy.spin()
