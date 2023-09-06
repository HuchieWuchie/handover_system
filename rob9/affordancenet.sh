#!/bin/bash
# You can run this script by running the following command in your terminal - "source connect.sh"
# DO NOT CHANGE MASTER_URI, only change ROS_IP to the IP of your computer
docker run --name affordancenet-ROS -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix --net=host --gpus all --rm huchiewuchie/affordancenet-ros /bin/bash
