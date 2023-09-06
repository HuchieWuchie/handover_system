//sudo chmod a+rw /dev/ttyACM0
  //rosrun urg_node urg_node

//roslaunch urdf_tutorial display.launch model:=/home/daniel/iiwa_ws/src/ROB10/rob10/urdf/iiwa_setup.urdf.xacro


//SOURCE - https://answers.ros.org/question/304562/x-y-and-z-coordinates-in-laserscan/

#include <iostream>
#include <sstream>
#include <unistd.h>
#include <algorithm>
#include <cmath>

#include "ros/ros.h"
#include "tf2_ros/static_transform_broadcaster.h"
#include "tf2/LinearMath/Quaternion.h"
 #include <tf/transform_listener.h>
#include "sensor_msgs/LaserScan.h"
#include "geometry_msgs/PointStamped.h"

#include "location_service/requestReceiverPose.h"

class ScannerServer{
  public:
    ScannerServer(){
      sub_scan = n.subscribe("scan", 1, &ScannerServer::scanCallback, this);
      pub_processed_scan =  n.advertise<sensor_msgs::LaserScan>("processed_scan", 1);
      pub_receiver_point = n.advertise<geometry_msgs::PointStamped>("receiver", 1);
      serverRequestReceiverPose = n.advertiseService("requestReceiverPose", &ScannerServer::getReceiverPose, this);
    }
    bool receiver_detected;

    void publishReceiverPose(){
      receiver_world.header.stamp = ros::Time::now();
      receiver_world.header.frame_id = "world";
      pub_receiver_point.publish(receiver_world);
    }


  private:
    ros::NodeHandle n;
    ros::ServiceServer serverRequestReceiverPose;
    ros::Subscriber sub_scan;
    ros::Publisher pub_processed_scan;
    ros::Publisher pub_receiver_point;
    tf2_ros::StaticTransformBroadcaster tf_broadcaster;
    std::vector<float> scan;
    float angle_increment;
    float angle_min;
    geometry_msgs::PointStamped receiver_laser;
    geometry_msgs::PointStamped receiver_world;
    geometry_msgs::PointStamped receiver_giver;
    tf::TransformListener listener;
    sensor_msgs::LaserScan processed_scan;
    std::vector<float> filtered_scan;

    void scanCallback(const sensor_msgs::LaserScan::ConstPtr& msg){
      //std::cout<<"callback"<<std::endl;
      scan = msg->ranges;
      angle_min = msg->angle_min;
      angle_increment = msg->angle_increment;


      processed_scan = *msg;
      processed_scan.ranges.clear();
      if (filtered_scan.size() != 0){
        processed_scan.header.stamp = ros::Time::now();
        processed_scan.ranges = filtered_scan;
        pub_processed_scan.publish(processed_scan);
      }
    }

    float calculate_median(std::vector<float>& v){
      int size = v.size();
      std::sort(v.begin(), v.end());
      float median;
      if (size % 2 != 0)
        median = v[size/2];
      else{
        median = (v[(size-1)/2] + v[size/2]) / 2.0;
        }
     return median;
   }

    void add_giver_frame(geometry_msgs::PointStamped& p){
     // TRANSFORM THE POINT TO THE LINK_0 FRAME
     geometry_msgs::PointStamped laser_point, iiwa_link_0_point;
     //laser_point.header.frame_id = "laser";
     //laser_point.header.stamp = ros::Time();
     //laser_point.point = p.point;

     try{
       //listener.transformPoint("iiwa_link_0", laser_point, iiwa_link_0_point);
       listener.transformPoint("iiwa_link_0", p, iiwa_link_0_point);
     }
     catch(tf::TransformException& ex){
       ROS_ERROR("Received an exception trying to transform a point from \"laser\" to \"iiwa_link_0\": %s", ex.what());
     }


     geometry_msgs::Point p_transformed;
     p_transformed = iiwa_link_0_point.point;
     //p_transformed.x = p.x + 0.332;
     //p_transformed.y = p.y + 0.179;

     // CALCULATE ANGLE
     float hypotenuse = sqrt(pow(p_transformed.x, 2.0f) + pow(p_transformed.y, 2.0f));
     float rot = asin(p_transformed.y/hypotenuse);
     //std::cout<<"rot "<<rot<<std::endl;

     // PUBLISH NEW FRAME
     geometry_msgs::TransformStamped giver_frame;
     giver_frame.header.stamp = ros::Time::now();
     giver_frame.header.frame_id = "iiwa_link_0";
     giver_frame.child_frame_id = "giver";
     giver_frame.transform.translation.x = 0.0f;
     giver_frame.transform.translation.y = 0.0f;
     giver_frame.transform.translation.z = 0.0f;
     tf2::Quaternion quat;
     quat.setRPY(0, 0, rot);
     giver_frame.transform.rotation.x = quat.x();
     giver_frame.transform.rotation.y = quat.y();
     giver_frame.transform.rotation.z = quat.z();
     giver_frame.transform.rotation.w = quat.w();
     tf_broadcaster.sendTransform(giver_frame);
   }

    bool getReceiverPose(location_service::requestReceiverPose::Request& req, location_service::requestReceiverPose::Response& res){
      std::cout<<"server"<<std::endl;
      filtered_scan.clear();
      std::vector<float> range;
      std::vector<float> angle_index;
      for (size_t i = 0; i < scan.size(); i++) {
        if ( scan[i]<0.5 || scan[i]>1.5 || std::isnan(scan[i]) ) {
          filtered_scan.push_back(0);
          continue;
        } else {
          filtered_scan.push_back(scan[i]);
          range.push_back(scan[i]);
          angle_index.push_back(i);
        }
      }
      //std::cout<<range.size()<<std::endl;

      if (range.size() == 0){
        receiver_detected = false;
        std::cout<<"No scan data left after filterring"<<std::endl;
        res.success.data = false;
        res.receiver.x = -1.0f;
        res.receiver.y = -1.0f;
        res.receiver.z = -1.0f;
      } else {
        receiver_detected = true;
        res.success.data = true;
        float median_range = calculate_median(range);
        float median_angle = calculate_median(angle_index);
        float angle = angle_min + (median_angle*angle_increment);
        //std::cout<<"Median_range "<<median_range<<std::endl;
        //std::cout<<"Median_angle "<<median_angle<<std::endl;
        //std::cout<<"angle "<<angle<<std::endl;

        // POLAR TO CARTESIAN + TRANSFORM TO WORLD FRAME
        receiver_laser.header.frame_id = "laser";
        receiver_laser.header.stamp = ros::Time();
        receiver_laser.point.x = median_range * cos(angle);
        receiver_laser.point.y = median_range * sin(angle);
        receiver_laser.point.z = 0.4f;
        //res.receiver = receiver.point;
        //res.receiver.x = median_range * cos(angle) + 0.43;
        //res.receiver.y = median_range * sin(angle) - 0.02;
        //res.receiver.z = 1.3f;

        add_giver_frame(receiver_laser);
        std::cout<<"Added a giver frame"<<std::endl;
        sleep(1);
        try{
          listener.transformPoint("world", receiver_laser, receiver_world);
          listener.transformPoint("giver", receiver_laser, receiver_giver);
        }
        catch(tf::TransformException& ex){
          ROS_ERROR("Received an exception trying to transform a point from \"laser\" to \"iiwa_link_0\": %s", ex.what());
        }

        std::cout<<"receiver_giver"<<receiver_giver<<std::endl;
        std::cout<<"================="<<std::endl;
        std::cout<<"receiver_world"<<receiver_world<<std::endl;
        //res.receiver = receiver_world.point;
        res.receiver = receiver_giver.point;
        //pub_processed_scan.publish(processed_scan);
      }
    }
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "scan_processing_server_node");
  ScannerServer server;
  server.receiver_detected = false;
  //ros::Rate loop_rate(30);

  std::cout<<"Scan processing server is running"<<std::endl;
  while(ros::ok()){
    if (server.receiver_detected){
      server.publishReceiverPose();
    }
    ros::spinOnce();
    //loop_rate.sleep();
  }

  return 0;
}

/*
rosrun tf tf_echo /world /laser
At time 0.000
- Translation: [0.430, -0.020, 0.975]
- Rotation: in Quaternion [0.000, 0.000, 0.000, 1.000]
            in RPY (radian) [0.000, -0.000, 0.000]
            in RPY (degree) [0.000, -0.000, 0.000]
========================================================
rosrun tf tf_echo /iiwa_link_0 /world
At time 0.000
- Translation: [0.332, 0.179, -1.038]
- Rotation: in Quaternion [0.000, 0.000, 0.000, 1.000]
            in RPY (radian) [0.000, -0.000, 0.000]
            in RPY (degree) [0.000, -0.000, 0.000]

========================================================
rosrun tf tf_echo /iiwa_link_0 /laser
At time 0.000
- Translation: [0.762, 0.159, -0.063]
- Rotation: in Quaternion [0.000, 0.000, 0.000, 1.000]
           in RPY (radian) [0.000, -0.000, 0.000]
           in RPY (degree) [0.000, -0.000, 0.000]

*/
