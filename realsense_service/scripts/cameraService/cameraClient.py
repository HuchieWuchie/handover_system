import sys
import rospy
from realsense_service.srv import *
import numpy as np
import cv2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header, Float32MultiArray, MultiArrayLayout, MultiArrayDimension
from sensor_msgs.msg import Image, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

class CameraClient(object):
    """docstring for CameraClient."""

    def __init__(self, type="realsenseD435"):
        self.type = type
        self.rgb = 0
        self.depth = 0
        self.uv = 0
        self.pointcloud = 0
        self.pointcloudColor = 0

        if self.type == "realsenseD435":
            self.baseService = "/sensors/realsense"
        else:
            raise Exception("Invalid type")

        self.serviceNameCapture = self.baseService + "/capture"
        self.serviceNameRGB = self.baseService + "/rgb"
        self.serviceNameDepth = self.baseService + "/depth"
        self.serviceNameUV = self.baseService + "/pointcloud/static/uv"
        self.serviceNamePointcloud = self.baseService + "/pointcloud/static"

    def captureNewScene(self):
        """ Tells the camera service to update the static data """

        rospy.wait_for_service(self.serviceNameCapture)
        captureService = rospy.ServiceProxy(self.serviceNameCapture, capture)
        msg = capture()
        msg.data = True
        response = captureService(msg)

    def getRGB(self):
        """ Sets the self.rgb to current static rgb captured by camera """

        rospy.wait_for_service(self.serviceNameRGB)
        rgbService = rospy.ServiceProxy(self.serviceNameRGB, rgb)
        msg = rgb()
        msg.data = True
        response = rgbService(msg)
        img = np.frombuffer(response.img.data, dtype=np.uint8).reshape(response.img.height, response.img.width, -1)
        self.rgb = img
        return self.rgb


    def getDepth(self):
        """ Sets the self.depth to current static depth image captured by
        camera """

        rospy.wait_for_service(self.serviceNameDepth)
        depthService = rospy.ServiceProxy(self.serviceNameDepth, depth)
        msg = depth()
        msg.data = True
        response = depthService(msg)
        img = np.frombuffer(response.img.data, dtype=np.float16).reshape(response.img.height, response.img.width, -1)
        #img = np.frombuffer(response.img.data, dtype=np.uint8).reshape(response.img.height, response.img.width, -1)
        self.depth = img
        return self.depth

    def getUvStatic(self):
        """ Sets the self.uv to current static uv coordinates for translation
        from pixel coordinates to point cloud coordinates """

        rospy.wait_for_service(self.serviceNameUV)
        uvStaticService = rospy.ServiceProxy(self.serviceNameUV, uvSrv)
        msg = uvSrv()

        msg.data = True
        response = uvStaticService(msg)
        uv = self.unpackUV(response.uv)

        self.uv = uv
        return self.uv

    def getPointCloudStatic(self):
        """ sets self.pointcloud to the current static point cloud with geometry
        only """

        rospy.wait_for_service(self.serviceNamePointcloud)
        pointcloudStaticService = rospy.ServiceProxy(self.serviceNamePointcloud, pointcloud)

        msg = pointcloud()
        msg.data = True
        response = pointcloudStaticService(msg)
        self.pointcloud, self.pointcloudColor = self.unpackPCD(response.pc, response.color)

        return self.pointcloud, self.pointcloudColor

    def packPCD(self, geometry, color):
        """ Input:
            geometry    - np.array, contains x, y, z information
            color       - np.array, contains r, g, b information

            Output:
            msg_geometry - sensor_msgs.msg.PointCloud2
            msg_color    - std_msgs.Float32MultiArray
        """
        FIELDS_XYZ = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "ptu_camera_color_optical_frame"

        msg_geometry = pc2.create_cloud(header, FIELDS_XYZ, geometry)
        msg_color = 0
        if color is not None:
            msg_color = Float32MultiArray()
            msg_color.data = color.astype(float).flatten().tolist()

        return msg_geometry, msg_color

    def unpackPCD(self, msg_geometry, msg_color):
        """ Input:
            msg_geometry - sensor_msgs.msg.PointCloud2
            msg_color    - std_msgs.Float32MultiArray

            Output:
            geometry    - np.array, contains x, y, z information
            color       - np.array, contains r, g, b information
        """

        # Get cloud data from ros_cloud
        field_names = [field.name for field in msg_geometry.fields]
        geometry_data = list(pc2.read_points(msg_geometry, skip_nans=True, field_names = field_names))

        # Check empty
        if len(geometry_data)==0:
            print("Converting an empty cloud")
            return None, None

        geometry = [(x, y, z) for x, y, z in geometry_data ] # get xyz
        geometry = np.array(geometry)

        # get colors
        color = 0
        if msg_color is not None:
            color = np.asarray(msg_color.data)
            color = np.reshape(color, (-1,3)) / 255
            color = np.flip(color, axis=1)

        return geometry, color


    def packUV(self, uv):
        """ packs the uv data into a Float32MultiArray

            Input:
            uv          -   np.array, int, shape (N, 2)

            Output:
            msg         -   std_msgs.msg.Float32MultiArray()
        """

        uvDim1 = MultiArrayDimension()
        uvDim1.label = "length"
        uvDim1.size = int(uv.shape[0] * uv.shape[1])
        uvDim1.stride = uv.shape[0]

        uvDim2 = MultiArrayDimension()
        uvDim2.label = "pair"
        uvDim2.size = uv.shape[1]
        uvDim2.stride = uv.shape[1]

        uvLayout = MultiArrayLayout()
        uvLayout.dim.append(uvDim1)
        uvLayout.dim.append(uvDim2)

        msg = Float32MultiArray()
        msg.data = uv.flatten().tolist()
        msg.layout = uvLayout

        return msg

    def unpackUV(self, msg):
        """ unpacks the uv msg into a numpy array

            Input:
            msg         -   std_msgs.msg.Float32MultiArray()

            Output:
            uv          -   np.array, int, shape (N, 2)
        """

        rows = int(msg.layout.dim[0].size / msg.layout.dim[1].size)
        cols = int(msg.layout.dim[1].size)

        uv = np.array(msg.data).astype(int)
        uv = np.reshape(uv, (rows, cols))
        return uv
