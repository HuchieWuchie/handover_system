#!/usr/bin/env python3


# Software License Agreement (BSD License)
#
# Copyright (c) 2012, Robotiq, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Robotiq, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2012, Robotiq, Inc.
# Revision $Id$

"""@package docstring
Command-line interface for sending simple commands to a ROS node controlling a 3F gripper gripper.

This serves as an example for publishing messages on the 'Robotiq3FGripperRobotOutput' topic using the 'Robotiq3FGripper_robot_output' msg type for sending commands to a 3F gripper gripper. In this example, only the simple control mode is implemented. For using the advanced control mode, please refer to the Robotiq support website (support.robotiq.com).
"""

# from __future__ import print_function
# import roslib;
# roslib.load_manifest('robotiq_3f_gripper_control')

import rospy
from robotiq_3f_gripper_articulated_msgs.msg import Robotiq3FGripperRobotOutput
from std_msgs.msg import Int16
from datetime import datetime

def genCommand(int, command):
    # reset
    if int == 0:
        command = Robotiq3FGripperRobotOutput();
        command.rACT = 0
    # activate
    if int == 1:
        command = Robotiq3FGripperRobotOutput();
        command.rACT = 1
        command.rGTO = 1
        command.rSPA = 255
        command.rFRA = 150
    # close
    if int == 2:
        command.rPRA = 255
    # open
    if int == 3:
        command.rPRA = 0
    # basic grip
    if int == 4:
        command.rMOD = 0
    # pinch
    if int == 5:
        command.rMOD = 1
    # wide grip
    if int == 6:
        command.rMOD = 2
    # scissors
    if int == 7:
        command.rMOD = 3

    # increase speed
    if int == 10:
        command.rSPA += 25
        if command.rSPA > 255:
            command.rSPA = 255

    # decrease speed
    if int == 20:
        command.rSPA -= 25
        if command.rSPA < 0:
            command.rSPA = 0

    # increase force
    if int == 30:
        command.rFRA += 25
        if command.rFRA > 255:
            command.rFRA = 255

    # decrease force
    if int == 40:
        command.rFRA -= 25
        if command.rFRA < 0:
            command.rFRA = 0

    # set the openess, valid values from 100 to 355
    if int > 99:
        try:
            command.rPRA = int - 100
            if command.rPRA > 255:
                command.rPRA = 255
            if command.rPRA < 0:
                command.rPRA = 0
        except ValueError:
            pass

    return command

def callback(msg):
    global command
    input = msg.data
    command = genCommand(input, command)
    dateTimeObj = datetime.now()
    print(str(dateTimeObj) + " msg " + str(msg))
    #print(command)
    pub.publish(command)


if __name__ == '__main__':
    rospy.init_node('gripper_controller')
    rospy.Subscriber('gripper_controller', Int16, callback)
    pub = rospy.Publisher('Robotiq3FGripperRobotOutput', Robotiq3FGripperRobotOutput, queue_size = 10)
    command = Robotiq3FGripperRobotOutput()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
