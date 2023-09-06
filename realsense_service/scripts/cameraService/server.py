#!/usr/bin/env python3
from __future__ import print_function

from realsense_service.srv import intrinsics, intrinsicsResponse
from realsense_service.srv import capture, captureResponse
from realsense_service.srv import depth, depthResponse
from realsense_service.srv import rgb, rgbResponse
from realsense_service.srv import uvSrv, uvSrvResponse
from realsense_service.srv import pointcloud, pointcloudResponse
from std_msgs.msg import Header, Float32MultiArray, MultiArrayLayout, MultiArrayDimension
from sensor_msgs.msg import Image, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import rospy
import pyrealsense2 as rs
import numpy as np
from cv_bridge import CvBridge
import cv2
import time
from ctypes import * # convert float to uint32
import time

cam_width = 1280
cam_height = 720

#cam_width = 640
#cam_height = 360


class RealsenseServer(object):
    """docstring for CameraServer."""

    def __init__(self, cam_width, cam_height, type="realsenseD435", tempFilterSize = 10):

        self.cam_width = cam_width
        self.cam_height = cam_height
        self.tempFilterSize = tempFilterSize
        self.type = type

        # captured information variables initialized to 0 or empty
        self.color_frame = 0
        self.color_image = 0
        self.depth_frame = 0
        self.depth_image = 0
        self.cloudGeometry = 0
        self.cloudColor = 0
        self.cloudGeometryStatic = 0
        self.cloudColorStatic = 0
        self.colorImageStatic = 0
        self.depthImageStatic = 0
        self.uv = 0
        self.uvStatic = 0
        self.framesRGB = []
        self.framesDepth = []

        self.br = CvBridge()

        self.FIELDS_XYZ = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        if self.type == "realsenseD435":
            self.baseService = "/sensors/realsense"
            self.nodeName = 'realsense_service'
        else:
            raise Exception("Invalid type")

        # Initialize ROS
        rospy.init_node(self.nodeName)

        self.serviceCapture = rospy.Service(self.baseService + '/capture', capture, self.updateStatic)
        self.serviceCaptureDepth = rospy.Service(self.baseService + '/depth', depth, self.serviceSendDepthImageStatic)
        self.serviceCaptureRGB = rospy.Service(self.baseService + '/rgb', rgb, self.serviceSendRGBImageStatic)
        self.serviceUVStatic = rospy.Service(self.baseService + '/pointcloud/static/uv', uvSrv, self.serviceUVStatic)
        self.servicePointCloudStatic = rospy.Service(self.baseService + '/pointcloud/static', pointcloud, self.servicePointCloud)


        self.pubPointCloudGeometryStatic = rospy.Publisher(self.baseService + "/pointcloudGeometry/static", PointCloud2, queue_size=1)
        self.pubStaticRGB = rospy.Publisher(self.baseService + "/rgb/static", Image, queue_size=1)
        self.pubStaticDepth = rospy.Publisher(self.baseService + "/depth/static", Image, queue_size = 1)
        self.pubPointCloudGeometryStaticRGB = rospy.Publisher(self.baseService + "/pointcloudGeometry/static/rgb", Float32MultiArray, queue_size=1)
        self.rate = rospy.Rate(5) #change back to 6
        self.frame_no = 0

        # Initialize realsense package
        self.initializeRealsense()

    def initializeRealsense(self):

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))


        self.config.enable_stream(rs.stream.color, self.cam_width, self.cam_height, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, self.cam_width, self.cam_height, rs.format.z16, 30)
        self.profile = self.pipeline.start(self.config)

        self.depth_sensor = self.profile.get_device().first_depth_sensor()

        preset_range = self.depth_sensor.get_option_range(rs.option.visual_preset)
        for i in range(int(preset_range.max)):
            visualpreset = self.depth_sensor.get_option_value_description(rs.option.visual_preset,i)
            if visualpreset == "High Accuracy":
                self.depth_sensor.set_option(rs.option.visual_preset, i)
                print("Setting realsense profile to: High Accuracy")

        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        rospy.sleep(3)
        self.update()

        # Set static information to initial captured information
        self.colorImageStatic = self.color_image
        self.depthImageStatic = self.depth_image
        self.cloudGeometryStatic = self.cloudGeometry
        self.cloudColorStatic = self.cloudColor
        self.uvStatic = self.uv

        self.intrinsics = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()


    def update(self, save_object = None, capture = False):
        """ Capture a new frame and updates frames and point clouds """

        self.frame = self.pipeline.wait_for_frames()
        self.aligned_frames = self.align.process(self.frame)

        if self.frame_no == 0 or capture == True:

            hole_filter = rs.hole_filling_filter(mode = 2)
            dec_filter = rs.decimation_filter()

            self.color_frame = self.aligned_frames.get_color_frame()
            self.color_image = np.asanyarray(self.color_frame.get_data())
            self.depth_frame = hole_filter.process(dec_filter.process(self.aligned_frames.get_depth_frame()))
            self.depth_image = np.asanyarray(self.depth_frame.get_data())


            """
            self.framesRGB.append(self.aligned_frames.get_color_frame())
            self.framesDepth.append(hole_filter.process(dec_filter.process(self.aligned_frames.get_depth_frame())))

            if len(self.framesRGB) > self.tempFilterSize:
                self.framesRGB.pop(0)
            if len(self.framesDepth) > self.tempFilterSize:
                self.framesDepth.pop(0)
            """


            #self.color_frame, self.depth_frame, self.color_image, self.depth_image = self.temporalFilter(self.framesRGB, self.framesDepth)
            self.cloudGeometry, self.cloudColor, self.uv = self.generatePointcloud(depth_frame = self.depth_frame, color_frame = self.color_frame, color_image = self.color_image, maxDistanceMeters = 1, maxVertices = 40000)

        if save_object != None:
            color_image = np.asanyarray(self.aligned_frames.get_color_frame().get_data())
        #    #print(self.color_image)
            save_object.write(color_image)

        self.frame_no += 1
    def temporalFilter(self, framesRGB, framesDepth):
        """ Filters captured information using pyrealsense temporal filter """

        temporal = rs.temporal_filter()

        for x in range(len(framesRGB)):
            filteredRGBFrame = temporal.process(framesRGB[x])

        for x in range(len(framesDepth)):
            filteredDepthFrame = temporal.process(framesDepth[x])

        # Convert images to numpy arrays
        filteredColor_image = np.asanyarray(filteredRGBFrame.get_data())
        filteredDepth_image = np.asanyarray(filteredDepthFrame.get_data())

        return filteredRGBFrame, filteredDepthFrame, filteredColor_image, filteredDepth_image


    def generatePointcloud(self, depth_frame, color_frame, color_image, maxDistanceMeters = 0.5, maxVertices = 80000):
        """ Generate point cloud and update the dynamic point clouds """

        cloud = rs.pointcloud()
        cloud.map_to(color_frame)
        points = rs.points()
        points = cloud.calculate(depth_frame)
        cloud = np.array(np.array(points.get_vertices()).tolist())

        uv = np.array(np.array(points.get_texture_coordinates()).tolist())
        uv[:,0] = uv[:,0] * self.cam_width
        uv[:,1] = uv[:,1] * self.cam_height
        uv[:, [1, 0]] = uv[:, [0, 1]]
        uv = np.rint(np.array(uv)).astype(int)

        idx = cloud[:,2] < maxDistanceMeters
        cloud = cloud[idx]
        uv = uv[idx]
        idxs = np.random.choice(cloud.shape[0], maxVertices, replace=False)
        cloudGeometry = cloud[idxs]
        uv = uv[idxs]

        colors = []
        for idx in uv:
            colors.append(color_image[idx[0], idx[1]])
        colors = np.array(colors)

        cloudColor = colors.flatten()

        return cloudGeometry, cloudColor, uv

    def sendIntrinsics(self):
        todo = True

    def serviceSendDepthImageStatic(self, command):
        br = CvBridge()
        return br.cv2_to_imgmsg(self.depthImageStatic)

    def serviceSendRGBImageStatic(self, command):
        br = CvBridge()
        return br.cv2_to_imgmsg(self.colorImageStatic)

    def serviceUVStatic(self, command):

        uvDim1 = MultiArrayDimension()
        uvDim1.label = "length"
        uvDim1.size = int(self.uvStatic.shape[0] * self.uvStatic.shape[1])
        uvDim1.stride = self.uvStatic.shape[0]

        uvDim2 = MultiArrayDimension()
        uvDim2.label = "pair"
        uvDim2.size = self.uvStatic.shape[1]
        uvDim2.stride = self.uvStatic.shape[1]

        uvLayout = MultiArrayLayout()
        uvLayout.dim.append(uvDim1)
        uvLayout.dim.append(uvDim2)

        uvMsg = Float32MultiArray()
        uvMsg.data = self.uvStatic.flatten().tolist()
        uvMsg.layout = uvLayout

        msg = uvSrvResponse()
        msg.uv = uvMsg
        return msg

    def servicePointCloud(self, command):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "ptu_camera_color_optical_frame"
        msg = pointcloudResponse()
        msg.pc = pc2.create_cloud(header, self.FIELDS_XYZ, self.cloudGeometryStatic)
        msg.color.data = camera.cloudColorStatic
        return msg

    def updateStatic(self, capture):
        """ Sets static information to latest images and point clouds captured """
        print("Setting new statics...")
        try:
            self.update(capture = True)
            self.colorImageStatic = self.color_image
            self.depthImageStatic = self.depth_image
            self.cloudGeometryStatic = self.cloudGeometry
            self.cloudColorStatic = self.cloudColor
            self.uvStatic = self.uv

            msg = captureResponse()
            msg.success.data = True
            return msg

        except:
            msg = captureResponse()
            msg.success.data = False
            return msg

if __name__ == "__main__":
    camera = RealsenseServer(cam_width = cam_width, cam_height = cam_height, type="realsenseD435", tempFilterSize=6)

    # We will continuously update the dynamic images with the latest information
    # captured from realsense
    #out_video = cv2.VideoWriter("realsense_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps=30, frameSize=(cam_width, cam_height), isColor=True)
    out_video = None
    br = CvBridge()
    while not rospy.is_shutdown():

        camera.update(save_object = out_video)
        print("Sending: ", rospy.Time.now())

        # publish static
        #camera.pubStaticRGB.publish(br.cv2_to_imgmsg(camera.colorImageStatic))

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "ptu_camera_color_optical_frame"
        camera.pubPointCloudGeometryStatic.publish(pc2.create_cloud(header, camera.FIELDS_XYZ, camera.cloudGeometryStatic))

        #msg = Float32MultiArray()
        #msg.data = camera.cloudColorStatic
        #camera.pubPointCloudGeometryStaticRGB.publish(msg)

        #msg = Float32MultiArray()
        #msg.data = camera.uvStatic
        #camera.pubPointCloudGeometryStaticIndex.publish(msg)


        camera.rate.sleep()
    out_video.close()
