#!/usr/bin/env python3
import rospy
from locationService.client import LocationClient


if __name__ == '__main__':
    rospy.init_node('scan_processing_example', anonymous=True)

    locClient = LocationClient()

    location = locClient.getLocation()

    print(location)
