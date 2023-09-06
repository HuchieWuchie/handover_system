#!/usr/bin/env python3
import numpy as np
import math

import rospy
import geometry_msgs
from geometry_msgs.msg import PoseStamped, Pose, Quaternion, Vector3, Transform
from std_msgs.msg import String

from rob9.srv import tf2TransformPoseStampedSrv, tf2TransformPoseStampedSrvResponse
from rob9.srv import tf2TransformPathSrv, tf2TransformPathSrvResponse
from rob9.srv import tf2GetTransformSrv, tf2GetTransformSrvResponse
from rob9.srv import tf2VisualizeTransformSrv, tf2VisualizeTransformSrvResponse

def visualizeTransform(transform, name):
    """ Input:
        transform           - geometry_msgs.Transform()
        name                - string, name of transform
    """

    rospy.wait_for_service("/tf2/visualize_transform")
    tf2Service = rospy.ServiceProxy("/tf2/visualize_transform", tf2VisualizeTransformSrv)

    _ = tf2Service(transform, String(name))


def transformToFrame(pose, newFrame, currentFrame = "ptu_camera_color_optical_frame"):
    """ input:  pose - geometry_msgs.PoseStamped()
                        numpy array (x, y, z)
                        numpy array (x, y, z, qx, qy, qz, qw)
                newFrame - desired frame for pose to be transformed into.
        output: transformed_pose_msg - pose in newFrame """

    if isinstance(pose, (np.ndarray, np.generic) ):
        npArr = pose

        pose = PoseStamped()
        pose.pose.position.x = npArr[0]
        pose.pose.position.y = npArr[1]
        pose.pose.position.z = npArr[2]

        if npArr.shape[0] < 4: # no orientation provided:
            pose.pose.orientation.w = 1
            pose.pose.orientation.x = 0
            pose.pose.orientation.y = 0
            pose.pose.orientation.z = 0

        else:
            pose.pose.orientation.w = npArr[6]
            pose.pose.orientation.x = npArr[3]
            pose.pose.orientation.y = npArr[4]
            pose.pose.orientation.z = npArr[5]
        pose.header.frame_id = currentFrame


    pose.header.stamp = rospy.Time.now()

    rospy.wait_for_service("/tf2/transformPoseStamped")
    tf2Service = rospy.ServiceProxy("/tf2/transformPoseStamped", tf2TransformPoseStampedSrv)

    response = tf2Service(pose, String(newFrame)).data

    return response

def transformToFramePath(path, newFrame):
    """ input:  pose - nav_msgs.Path()
            newFrame - desired frame for pose to be transformed into.
    output: transformed_path_msg - path in newFrame """

    pose.header.stamp = rospy.Time.now()

    rospy.wait_for_service("/tf2/transformPath")
    tf2Service = rospy.ServiceProxy("/tf2/transformPath", tf2TransformPathSrv)

    response = tf2Service(path, String(newFrame))

    return response

def poseToTransform(pose):
    """ Input:
        pose                - array [x, y, z, qx, qy, qz, qw]

        Output:
        transform           - geometry_msgs.Transform()
    """

    transform = Transform()

    transform.translation.x = pose[0]
    transform.translation.y = pose[1]
    transform.translation.z = pose[2]

    transform.rotation.x = pose[3]
    transform.rotation.y = pose[4]
    transform.rotation.z = pose[5]
    transform.rotation.w = pose[6]

    return transform

def poseMsgToTransformMsg(pose):
    """ Input:
        pose                - geometry_msgs.Pose()

        Output:
        transform           - geometry_msgs.Transform()
    """

    transform = Transform()

    transform.translation.x = pose.position.x
    transform.translation.y = pose.position.y
    transform.translation.z = pose.position.z

    transform.rotation.x = pose.orientation.x
    transform.rotation.y = pose.orientation.y
    transform.rotation.z = pose.orientation.z
    transform.rotation.w = pose.orientation.w

    return transform


def getTransform(source_frame, target_frame):
    """ input:
        source_frame    - string
        target_frame    - string

        output:
        T               - 4x4 homogeneous transformation, np.array()
        transl          - x, y, z translation, np.array(), shape (3)
        rot             - 3x3 rotation matrix, np.array(), shape (3, 3)
    """

    rospy.wait_for_service("/tf2/get_transform")
    tf2Service = rospy.ServiceProxy("/tf2/get_transform", tf2GetTransformSrv)

    source_msg = String()
    source_msg.data = source_frame
    target_msg = String()
    target_msg.data = target_frame

    response = tf2Service(source_msg, target_msg)

    transl = np.zeros((3, 1))
    transl[0] = response.transform.translation.x
    transl[1] = response.transform.translation.y
    transl[2] = response.transform.translation.z

    msg_quat = response.transform.rotation
    quat = [msg_quat.x, msg_quat.y, msg_quat.z, msg_quat.w]
    rot = quatToRot(quat)

    T = np.identity(4)
    T[0:3, 0:3] = rot
    T[0, 3] = response.transform.translation.x
    T[1, 3] = response.transform.translation.y
    T[2, 3] = response.transform.translation.z

    return T, transl.flatten(), rot

def quatToRot(q):
    """ input:  -   q, array [x, y, z, w]
        output: -   R, matrix 3x3 rotation matrix
        https://github.com/cgohlke/transformations/blob/master/transformations/transformations.py
    """

    x, y, z, w = q
    quaternion = [w, x, y, z]


    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < np.finfo(float).eps * 4.0:
        return np.identity(3).flatten()

    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    R = np.array(
        [
            [
                1.0 - q[2, 2] - q[3, 3],
                q[1, 2] - q[3, 0],
                q[1, 3] + q[2, 0],
                0.0,
            ],
            [
                q[1, 2] + q[3, 0],
                1.0 - q[1, 1] - q[3, 3],
                q[2, 3] - q[1, 0],
                0.0,
            ],
            [
                q[1, 3] - q[2, 0],
                q[2, 3] + q[1, 0],
                1.0 - q[1, 1] - q[2, 2],
                0.0,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    return R[:3, :3]


def cartesianToSpherical(x, y, z):
    """ input: cartesian coordinates
        output: 3 spherical coordinates """

    polar = math.atan2(math.sqrt(x**2 + y**2), z)
    azimuth = math.atan2(y, x)
    r = math.sqrt(x**2 + y**2 + z**2)
    return r, polar, azimuth

def quaternionMultiply(q1, q2):
    """ input:  -   q1, array or list, format xyzw
                -   q2, array or list, format xyzw
        output: -   q,  array or list, format xyzw
        https://github.com/cgohlke/transformations/blob/master/transformations/transformations.py
    """

    # confusing order but this is on purpose

    x2, y2, z2, w2 = q1
    x1, y1, z1, w1 = q2

    q = [
            -x2 * x1 - y2 * y1 - z2 * z1 + w2 * w1,
            x2 * w1 + y2 * z1 - z2 * y1 + w2 * x1,
            -x2 * z1 + y2 * w1 + z2 * x1 + w2 * y1,
            x2 * y1 - y2 * x1 + z2 * w1 + w2 * z1,
        ]

    x, y, z, w = q[1], q[2], q[3], q[0]

    return [x, y, z, w]

def quaternionConjugate(q):
    """ input:  -   q, array or list, format xyzw
        output: -   qc,  array or list, format xyzw
        https://github.com/cgohlke/transformations/blob/master/transformations/transformations.py
    """

    x, y, z, w = q
    qc = [-x, -y, -z, w]

    return qc

def unitVector(v):
    """ input:  -   v, array or list [x, y, z]
        output: -   v_norm,  array or list [x, y, z]
        normalizes a given vector using tf2_ros """

    v = np.array(v, dtype = np.float64, copy=True)
    if v.ndim == 1:
        v /= math.sqrt(np.dot(v, v))

    return v

def quaternionFromRotation(R):
    """ Input:  R   -   3 x 3 numpy matrix or 2D list, rotation matrix
        Output: q   -   1 x 4 numpy array, quaternion (x, y, z, w)
        https://github.com/cgohlke/transformations/blob/master/transformations/transformations.py
    """

    # symmetric matrix K
    K = np.array(
        [
            [R[0,0] - R[1,1] - R[2,2], 0.0, 0.0, 0.0],
            [R[0,1] + R[1,0], R[1,1] - R[0,0] - R[2,2], 0.0, 0.0],
            [R[0,2] + R[2,0], R[1,2] + R[2,1], R[2,2] - R[0,0] - R[1,1], 0.0],
            [R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1], R[0,0] + R[1,1] + R[2,2]],
        ]
    )
    K /= 3.0
    # quaternion is eigenvector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)

    #q = np.flip(q) # reverse the array to get [x, y, z, w]
    w, x, y, z = q
    q[0] = x
    q[1] = y
    q[2] = z
    q[3] = w

    return q

def poseToMatrix(pose):
    """ Input:
        pose         - array [x, y, z, qx, qy, qz, qw]

        Output:
        T           - np.array, shape (4,4) homogeneous transformation matrix
    """

    position = np.zeros((3))
    position[0] = pose[0]
    position[1] = pose[1]
    position[2] = pose[2]

    quaternion = np.zeros(4)
    quaternion[0] = pose[3]
    quaternion[1] = pose[4]
    quaternion[2] = pose[5]
    quaternion[3] = pose[6]

    rotMat = quatToRot(quaternion)

    T = np.identity(4)
    T[:3,:3] = rotMat
    T[:3,3] = position

    return T

def poseStampedToMatrix(msg):
    """ Input:
        msg         - geometry_msgs/PoseStamped

        Output:
        T           - np.array, shape (4,4) homogeneous transformation matrix
    """

    position = np.zeros((3))
    position[0] = msg.pose.position.x
    position[1] = msg.pose.position.y
    position[2] = msg.pose.position.z

    quaternion = np.zeros(4)
    quaternion[0] = msg.pose.orientation.x
    quaternion[1] = msg.pose.orientation.y
    quaternion[2] = msg.pose.orientation.z
    quaternion[3] = msg.pose.orientation.w

    rotMat = quatToRot(quaternion)

    T = np.identity(4)
    T[:3,:3] = rotMat
    T[:3,3] = position

    return T



def eulerFromQuaternion(q):
    pass
"""
def delta_orientation(pose1, pose2):
    # input: 2 posestamped ros message
    #    output: the difference in rotation expressed as a quaternion

    q1 = (
        pose1.pose.orientation.x,
        pose1.pose.orientation.y,
        pose1.pose.orientation.z,
        pose1.pose.orientation.w)
    rpy1 = np.asarray(euler_from_quaternion(q1)) * 180 / math.pi

    q2Inv = (
        pose2.transform.rotation.x,
        pose2.transform.rotation.y,
        pose2.transform.rotation.z,
        -pose2.transform.rotation.w)

    deltaQuaternion = quaternionMultiply(q1, q2Inv)
    #deltaRPY = np.asarray(euler_from_quaternion(deltaQuaternion)) * 180 / math.pi
    return deltaRPY
"""
if __name__ == '__main__':
    print("Start")
    R = quatToRot([0, 0, 0, 1])
    q = quatFromRotation(R)
    print("quaternion: ", q)

    q1 = [0, 0, 0, 1]
    print(quaternionConjugate(q1))

    v = [2, 5, 1]
    print(unitVector(v))

    q2 = [0, 1, 0, 0]
    q3 = [0.008, 0.23, 0.97, 0.89]
    print(quaternionMultiply(q2, q3))

    q = quaternionMultiply(q2,q3)
    print(quatToRot(q))
