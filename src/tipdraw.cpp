#include <ros/ros.h>
#include <gazebo_msgs/LinkStates.h>

#include <ignition/transport.hh>
#include <ignition/math.hh>
#include <ignition/msgs.hh>
#include <gazebo/common/Time.hh>

#include <Eigen/Dense>

ignition::transport::Node node;

// void counterCallback(const project::DrawingTip::ConstPtr &msg)
// {
//   ignition::msgs::Marker markerMsg;
//   markerMsg.set_ns("default");
//   markerMsg.set_id(0);
//   markerMsg.set_action(ignition::msgs::Marker::ADD_MODIFY);
//   markerMsg.set_type(ignition::msgs::Marker::SPHERE);

//   ignition::msgs::Material *matMsg = markerMsg.mutable_material();
//   matMsg->mutable_script()->set_name("Gazebo/BlueLaser");

//   ignition::msgs::Set(markerMsg.mutable_pose(),
//                       ignition::math::Pose3d(msg->x, msg->y, msg->z, 0, 0, 0));
//   ignition::msgs::Set(markerMsg.mutable_scale(),
//                       ignition::math::Vector3d(0.05, 0.05, 0.05));
//   node.Request("/marker", markerMsg);
// }

int getIndex(std::vector<std::string> v, std::string value)
{
  for (int i = 0; i < v.size(); i++)
  {
    if (v[i].compare(value) == 0)
      return i;
  }
  return -1;
}

void link_states_callback(gazebo_msgs::LinkStates link_states)
{
  int link7index = getIndex(link_states.name, "sevendof::link7");
  geometry_msgs::Pose pose = link_states.pose[link7index];

  ignition::msgs::Marker markerMsg;
  markerMsg.set_ns("default");
  markerMsg.set_id(0);
  markerMsg.set_action(ignition::msgs::Marker::ADD_MODIFY);
  markerMsg.set_type(ignition::msgs::Marker::SPHERE);

  ignition::msgs::Material *matMsg = markerMsg.mutable_material();
  matMsg->mutable_script()->set_name("Gazebo/BlueLaser");

  ignition::msgs::Set(markerMsg.mutable_pose(),
                      ignition::math::Pose3d(pose.position.x, pose.position.y, pose.position.z, 0, 0, 0));
  ignition::msgs::Set(markerMsg.mutable_scale(),
                      ignition::math::Vector3d(0.01, 0.01, 0.01));
  node.Request("/marker", markerMsg);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "tipdraw");
  ros::NodeHandle nh;

  ros::Subscriber tipsub = nh.subscribe("/gazebo/link_states", 100, link_states_callback);

  ros::spin();

  return 0;
}