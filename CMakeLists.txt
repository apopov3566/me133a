cmake_minimum_required(VERSION 2.8.3)
project(project)

## Set the package dependencies: Find catkin macros/libraries and
## other required packages.
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  geometry_msgs
  sensor_msgs
  std_msgs
  visualization_msgs
  urdfdom_py
  message_generation)

find_package(Eigen3 REQUIRED)
find_package(gazebo REQUIRED)

add_message_files(
  FILES
  DrawingTip.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)
## Uncomment if the package has a setup.py
# catkin_python_setup()

## Catkin configuration.  We don't have dependent projects.
catkin_package()

###########
## Build ##
###########

## Specify locations of header files
include_directories(${catkin_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})

## Declare the C++ executables
add_executable(tipdraw src/tipdraw.cpp)
add_dependencies(tipdraw ${tipdraw_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(tipdraw
  ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES}
)
## Add cmake target dependencies of the executables

## Specify libraries to link a library or executable target against
