#include <ros/ros.h>
#include <project/DrawingTip.h>

#include <ignition/transport.hh>
#include <ignition/math.hh>
#include <ignition/msgs.hh>
#include <gazebo/common/Time.hh>

// #include <Eigen/Dense>

ignition::transport::Node node;

void counterCallback(const project::DrawingTip::ConstPtr &msg)
{
  ignition::msgs::Marker markerMsg;
  markerMsg.set_ns("default");
  markerMsg.set_id(0);
  markerMsg.set_action(ignition::msgs::Marker::ADD_MODIFY);
  markerMsg.set_type(ignition::msgs::Marker::SPHERE);

  ignition::msgs::Material *matMsg = markerMsg.mutable_material();
  matMsg->mutable_script()->set_name("Gazebo/Black");

  ignition::msgs::Set(markerMsg.mutable_pose(),
                      ignition::math::Pose3d(msg->x, msg->y, msg->z, 0, 0, 0));
  ignition::msgs::Set(markerMsg.mutable_scale(),
                      ignition::math::Vector3d(0.02, 0.02, 0.02));
  // ignition::msgs::Time *lt = ignition::msgs::Time.New();
  // lt->set_sec(1);
  // ignition::msgs::Set(markerMsg.mutable_lifetime(), lt);
  node.Request("/marker", markerMsg);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "tipdraw");
  ros::NodeHandle nh;
  ros::Subscriber tipsub = nh.subscribe("/tipdraw", 10, counterCallback);
  ros::spin();
  return 0;
}
