#!/bin/bash
# You can run this script by running the following command in your terminal - "source connect.sh"
# DO NOT CHANGE MASTER_URI, only change ROS_IP to the IP of your computer
localIP="$(ip route get 8.8.8.8 | awk -F"src " 'NR==1{split($2,a," ");print a[1]}')"

export ROS_MASTER_URI="http://192.168.1.101:11311"
export ROS_IP="$localIP"
