#!/usr/bin/env python3
#
#   gazebodemo_trajectory.py
#
#   Create a motion, to send to Gazebo for the sevenDOF.
#
#   Publish:   /sevendof/j1_pd_control/command    std_msgs/Float64
#   Publish:   /sevendof/j2_pd_control/command    std_msgs/Float64
#   Publish:   /sevendof/j3_pd_control/command    std_msgs/Float64
#   Publish:   /sevendof/j4_pd_control/command    std_msgs/Float64
#   Publish:   /sevendof/j5_pd_control/command    std_msgs/Float64
#   Publish:   /sevendof/j6_pd_control/command    std_msgs/Float64
#   Publish:   /sevendof/j7_pd_control/command    std_msgs/Float64
#
import rospy
import numpy as np

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from urdf_parser_py.urdf import Robot
from draw_controller import getCanvases

from math import pi

# Import the kinematics stuff:
from kinematics import Kinematics, p_from_T, R_from_T, Rx, Ry, Rz

# We could also import the whole thing ("import kinematics"),
# but then we'd have to write "kinematics.p_from_T()" ...

# Import the Spline stuff:
from splines import CubicSpline, Goto, Hold, Stay, QuinticSpline, Goto5, LinearSpline


CANVAS_ATTRACTION_DISTANCE = 0.03  # distance when pen starts pull to canvas
CANVAS_TARGET_DEPTH = 0  # target pull depth (stops pulling when reached)
CANVAS_PRESSURE_SCALING = 0.3  # strength of pull based on distance

#
#  Generator Class
#


class Generator:
    # Initialize.
    def __init__(self):
        # The Gazebo controllers treat each joint seperately.  We thus
        # need a seperate publisher for each joint under the topic
        # "/BOTNAME/CONTROLLER/command"...
        self.N = 7
        self.pubs = []

        self.canvases = getCanvases()

        for i in range(self.N):
            topic = "/sevendof/j" + str(i + 1) + "_pd_control/command"
            self.pubs.append(rospy.Publisher(topic, Float64, queue_size=10))
        # # We used to add a short delay to allow the connection to form
        # # before we start sending anything.  However, if we start
        # # Gazebo "paused", this already provides time for the system
        # # to set up, before the clock starts.
        # rospy.sleep(0.25)

        # Find the simulation's starting position.  This will block,
        # but that's appropriate if we don't want to start until we
        # have this information.  Of course, the simulation starts at
        # zero, so we can simply use that information too.
        msg = rospy.wait_for_message("/sevendof/joint_states", JointState)
        self.theta_a = np.array(msg.position).reshape(7)
        self.theta_d = np.array(msg.position).reshape(7)
        rospy.loginfo("Gazebo's starting position: %s", str(self.theta_a))

        # IF we wanted to do kinematics:
        # # Grab the robot's URDF from the parameter server.
        robot = Robot.from_parameter_server()
        # # Instantiate the Kinematics
        self.kin = Kinematics(robot, 'world', 'tip')

        # Pick a starting and final joint position.
        p0r = np.array([0, 0.5, 0.3, 0]).reshape((4, 1))
        pA = np.array([-0.25, 0.697, 0.3, 0]).reshape((4, 1))
        pB = np.array([-0.25, 0.697, 0.8, 0]).reshape((4, 1))
        pC = np.array([0.25, 0.697, 0.8, 0]).reshape((4, 1))
        pD = np.array([0.25, 0.697, 0.3, 0]).reshape((4, 1))

        p1r = np.array([-0.5, 0, 0.3, pi/2]).reshape((4, 1))
        p2r = np.array([0, -0.5, 0.3, pi]).reshape((4, 1))

        pA2 = np.array([-0.25, -0.697, 0.3, pi]).reshape((4, 1))
        pB2 = np.array([-0.25, -0.697, 0.8, pi]).reshape((4, 1))
        pC2 = np.array([0.25, -0.697, 0.8, pi]).reshape((4, 1))
        pD2 = np.array([0.25, -0.697, 0.3, pi]).reshape((4, 1))

        T, J = self.kin.fkin(self.theta_a)
        p = p_from_T(T).reshape(3)
        p0 = np.array([[p[0], p[1], p[2], 0]]).reshape((4, 1))

        self.segments = [
            Hold(p0, 1.0),
            Goto(p0, p0r, 2.0),
            Goto(p0r, pA, 2.0),
            Hold(pA, 1.0),
            Goto(pA, pB, 2.0),
            Goto(pB, pC, 2.0),
            Goto(pC, pD, 2.0),
            Goto(pD, pA, 2.0),
            Goto(pA, p0r, 2.0),
            Goto(p0r, p1r, 2.0),
            Goto(p1r, p2r, 2.0),
            Goto(p2r, pA2, 2.0),
            Hold(pA2, 1.0),
            Goto(pA2, pB2, 2.0),
            Goto(pB2, pC2, 2.0),
            Goto(pC2, pD2, 2.0),
            Goto(pD2, pA2, 2.0),
            Goto(pA2, p2r, 2.0),
            Hold(p2r, 1000.0),
        ]

        # Also reset the trajectory, starting at the beginning.
        self.reset()

        rospy.Subscriber("/sevendof/joint_states",
                         JointState, self.jointCallback)

    def jointCallback(self, msg):
        self.theta_a = np.array(msg.position).reshape(7)
        # print("t", self.theta_a)

    # Reset.  If the simulation resets, also restart the trajectory.
    def reset(self):
        # Just reset the segment index counter and start time to zero.
        self.t0 = 0.0
        self.index = 0

    # Update is called every 10ms of simulation time!
    def update(self, t, dt):
        # If the current trajectory segment is done, shift to the next.
        dur = self.segments[self.index].duration()
        if t - self.t0 >= dur:
            self.t0 = self.t0 + dur
            # self.index = (self.index + 1)                       # not cyclic!
            self.index = (self.index + 1) % len(self.segments)  # cyclic!

        # Check whether we are done with all trajectory segments.
        if self.index >= len(self.segments):
            rospy.signal_shutdown("Done with motion")
            return

        # Grab the spline output as joint values.
        (p, pd) = self.segments[self.index].evaluate(t - self.t0)

        # print("p", p)
        # print("pd", pd)

        weights = np.eye(6)
        T, _ = self.kin.fkin(self.theta_a)
        if (p_from_T(T).reshape(3)[1] > 0.67):
            np.fill_diagonal(weights, [2, 0.05, 2, 5, 5, 5])
        else:
            np.fill_diagonal(weights, [2.0, 0.1, 2, 5, 5, 5])

        # if near a board, create pressure into the board
        pde = np.array([0.0, 0.0, 0.0])
        for C in self.canvases:
            pc = p_from_T(C @ T).reshape(3)
            if pc[1] < CANVAS_ATTRACTION_DISTANCE:
                pde += (np.linalg.inv(R_from_T(C)
                                      ) @ np.array([0.0, min(0.0, CANVAS_PRESSURE_SCALING * (CANVAS_TARGET_DEPTH - pc[1])), 0.0])).reshape(3)

        (self.theta_d, _) = self.kin.velocity_ikin_a(
            self.theta_d, self.theta_a, p[0:3, :].reshape((3, 1)), Rz(p[3]), pd[0:3, :].reshape((3, 1)) + pde.reshape((3, 1)), np.array([0, 0, pd[3]]).reshape((3, 1)), weights, 0.5, dt)

        # Send the individal angle commands.
        for i in range(self.N):
            self.pubs[i].publish(Float64(self.theta_d[i]))


#
#  Main Code
#
if __name__ == "__main__":
    # Prepare/initialize this node.
    rospy.init_node("trajectory")

    # Instantiate the trajectory generator object, encapsulating all
    # the computation and local variables.
    generator = Generator()

    # Prepare a servo loop at 100Hz.
    rate = 100
    servo = rospy.Rate(rate)
    dt = servo.sleep_dur.to_sec()
    rospy.loginfo(
        "Running the servo loop with dt of %f seconds (%fHz)" % (dt, rate))

    # Run the servo loop until shutdown (killed or ctrl-C'ed).  This
    # relies on rospy.Time, which is set by the simulation.  Therefore
    # slower-than-realtime simulations propagate correctly.
    starttime = rospy.Time.now()
    lasttime = starttime
    while not rospy.is_shutdown():

        # Current time (since start)
        servotime = rospy.Time.now()
        t = (servotime - starttime).to_sec()
        dt = (servotime - lasttime).to_sec()
        lasttime = servotime

        # Update the controller.
        generator.update(t, dt)

        # Wait for the next turn.  The timing is determined by the
        # above definition of servo.  Note, if you reset the
        # simulation, the time jumps back to zero and triggers an
        # exception.  If desired, we can simple reset the time here to
        # and start all over again.
        try:
            servo.sleep()
        except rospy.exceptions.ROSTimeMovedBackwardsException:
            # Reset the time counters, as well as the trajectory
            # generator object.
            rospy.loginfo("Resetting...")
            generator.reset()
            starttime = rospy.Time.now()
            lasttime = starttime
