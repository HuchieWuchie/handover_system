#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image

rospy.init_node('webcam_publisher', anonymous=True)
img_pub = rospy.Publisher('/sensors/webcam/img', Image, queue_size=1, latch=True)
rate = rospy.Rate(25)

vid_capture = cv2.VideoCapture(4)
#vid_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#vid_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 980)
print("Running webcam topic")
while not rospy.is_shutdown():
    # Capture each frame of webcam video
    ret,frame = vid_capture.read()

    img_msg = Image()
    img_msg.height = frame.shape[0]
    img_msg.width = frame.shape[1]
    img_msg.encoding = "rgb8"
    img_msg.data = frame.flatten().tolist()
    img_pub.publish(img_msg)

    #cv2.imshow("My cam video", frame)
    # Close and break the loop after pressing "x" key
    #if cv2.waitKey(1) &0XFF == ord('x'):
    #    break
    rate.sleep()
cv2.destroyAllWindows()
vid_capture.release()
