# Optimizing Robot-to-Human Object Handovers using Vision-based Affordance Information
Contains the code and implementaiton for our paper Optimizing Robot-to-Human Object Handovers using Vision-based Affordance Information.

## Description

This repository contains our ROS node implementation of "Optimizing Robot-to-Human Object Handovers using Vision-based Affordance Information". The system is capable of performing task-oriented handovers, where an object is grasped by its functional affordance and handover with an appropriate orientaiton. Object affordances are detected using our deep neural network AffNet-DR, which was trained solely on synthetic data.

## Handover orientation analysis
Scripts and code for deriving the mean handover orientation can be found in the folder named "handover_orientation_analysis" refer to the Readme file in there for more information.

## Affiliated links
For more information on how to train the object affordance segmentation deep neural network please see: https://github.com/HuchieWuchie/affnetDR

## Requirements:

General system requirements
```
CUDA version 11.6
NVIDIA GPU driver 510.60.02
ROS melodic
ros-melodic-moveit
ros-melodic-urg-node
```

C++:
```
realsense2
PCL (point cloud library)
OpenCV
```

Python 3.6.9
```
open3d 0.15.2
cv2 4.2.0
numpy 1.19.5
scipy 1.5.4
scikit_learn 0.24.2
torch (Pytorch) 1.10.2 cuda version
torchvision 0.11.2 cuda
scikit_image 0.17.2
PIL 8.4.0
rospkg 1.4.0
```

The system ran on a Lenovo Thinkpad P53 laptop with a Quadro RTX 4000 GPU with 8 GB VRAM and an Intel Core i9-9880H CPU 2.3 GHZ and 32 GB RAM.


## Installation:
```
mkdir ros_ws
mkdir ros_ws/src
cd ros_ws/src

git clone https://github.com/IFL-CAMP/iiwa_stack.git
git clone https://github.com/ros-industrial/robotiq.git
git clone https://github.com/HuchieWuchie/handover_system.git

cd ..
catkin_make
source devel/setup.bash
```

Download pretrained weights from: https://drive.google.com/file/d/1psCn_aT5KUyQDJrdxqR7GJgHeCewGokS/view?usp=sharing

Place and rename the weights file to ros_ws/src/affordanceAnalyzer/scripts/affordance_synthetic/weights.pth

## Setup the KUKA LBR iiwa 7 R800:

Install the iiwa stack found here: https://github.com/IFL-CAMP/iiwa_stack
Run the ROSSmartservo package on the KUKA controller, do this after launching ros on your ROS pc.

## Setup of the ROS pc:

Connect an ethernet cable between the ROS pc and the KUKA sunrise controller. Setup the network configuration on your ROS pc to the following:

```
IP: 172.31.1.150
Netmask: 255.255.0.0
```

Export ros settings 
```
export ROS_IP=172.31.1.150
export ROS_MASTER_URI=http://172.31.1.150:11311
```

Modify permission for the laser scanner
```
sudo chmod a+rw /dev/ttyACM0      # note that the usb port might change
```

## Usage 

launch roscore and launch file
```
source devel/setup.bash
roscore
roslaunch iiwa_noPtu_moveit moveit_planning_execution.launch
```

Launch whatever experiement you want, chose between the ones listed below.
```
rosrun rob10 final_test_observation.py
rosrun rob10 final_test_rule.py
rosrun rob10 orientation_test_observation.py # user study on orientation methods
rosrun rob10 orientation_test_rule.py
rosrun rob10 orientation_test_random.py
```

In order to command the robot to pick up an object you must send a command to the rostopic /objects_affordances_id. The integer id corresponds to the object classes of AffNet-DR, eg. 1 (knife), 16 (mallet), etc.

Note if you want to run the orientation_test_METHOD.py scripts you have to make use of precomputed information which can be found at: https://drive.google.com/file/d/1OhkOdDlKzmiacBYNIeN8ccKTg_f816GE/view?usp=sharing

## Authors
Daniel Lehotsky

Albert Christensen

Dimitrios Chrysostomou
