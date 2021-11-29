#!/usr/bin/env python3
#
#   gazebodemo_trajectory.py
#
#   Create a motion, to send to Gazebo for the twelvedof.
#
#   Publish:   /twelvedof/j1_pd_control/command    std_msgs/Float64
#   Publish:   /twelvedof/j2_pd_control/command    std_msgs/Float64
#   Publish:   /twelvedof/j3_pd_control/command    std_msgs/Float64
#   Publish:   /twelvedof/j4_pd_control/command    std_msgs/Float64
#   Publish:   /twelvedof/j5_pd_control/command    std_msgs/Float64
#   Publish:   /twelvedof/j6_pd_control/command    std_msgs/Float64
#   Publish:   /twelvedof/j7_pd_control/command    std_msgs/Float64
#   Publish:   /twelvedof/j8_pd_control/command    std_msgs/Float64
#   Publish:   /twelvedof/j9_pd_control/command    std_msgs/Float64
#   Publish:   /twelvedof/j10_pd_control/command    std_msgs/Float64
#   Publish:   /twelvedof/j11_pd_control/command    std_msgs/Float64
#   Publish:   /twelvedof/j12_pd_control/command    std_msgs/Float64
#
import rospy
import numpy as np

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from urdf_parser_py.urdf import Robot
from draw_controller import getCanvases

from math import pi

# Import the kinematics stuff:
from kinematics12 import Kinematics, p_from_T, R_from_T, Rx, Ry, Rz

# We could also import the whole thing ("import kinematics"),
# but then we'd have to write "kinematics.p_from_T()" ...

# Import the Spline stuff:
from splines import CubicSpline, Goto, Hold, Stay, QuinticSpline, Goto5, LinearSpline


CANVAS_ATTRACTION_DISTANCE = 0.07  # distance when pen starts pull to canvas
CANVAS_TARGET_DEPTH = 0  # target pull depth (stops pulling when reached)
CANVAS_PRESSURE_SCALING = 1.0  # strength of pull based on distance

#
#  Generator Class
#


class Image:
    def __init__(self, segments):
        self.segments = segments
        self.index = 0

        self.t0 = 0.0
        self.tstop = 0
        self.tstart = 0
        self.pause = False

    def pause(self, t):
        if not self.pause:
            self.pause = True
            self.tstop = t

    def unpause(self, t):
        if self.pause:
            self.pause = False
            self.tstart += t - self.tstop

    def update(self, t):
        te = t - self.tstart
        dur = self.segments[self.index].duration()
        if te - self.t0 >= dur:
            self.t0 = self.t0 + dur
            # self.index = (self.index + 1)                       # not cyclic!
            self.index = (self.index + 1) % len(self.segments)  # cyclic!

        # Check whether we are done with all trajectory segments.
        if self.index >= len(self.segments):
            rospy.signal_shutdown("Done with motion")
            return

        # Grab the spline output as joint values.
        return self.segments[self.index].evaluate(te - self.t0)


class Generator:
    # Initialize.
    def __init__(self):
        # The Gazebo controllers treat each joint seperately.  We thus
        # need a seperate publisher for each joint under the topic
        # "/BOTNAME/CONTROLLER/command"...
        self.N = 12
        self.pubs = []

        self.canvases = getCanvases()

        for i in range(self.N):
            topic = "/twelvedof/j" + str(i + 1) + "_pd_control/command"
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
        msg = rospy.wait_for_message("/twelvedof/joint_states", JointState)
        self.theta_a = np.array(msg.position).reshape(12)
        self.theta_d = np.array(msg.position).reshape(12)

        # IF we wanted to do kinematics:
        # # Grab the robot's URDF from the parameter server.
        robot = Robot.from_parameter_server()
        # # Instantiate the Kinematics

        self.kin = Kinematics(robot, "world", ["tip_full", "tip_elbow"])

        T, _ = self.kin.fkin(self.theta_d, 0)
        p = p_from_T(T).reshape(3)

        # Pick a starting and final joint position.
        p0 = np.array([[p[0], p[1], p[2], 0]]).reshape((4, 1))
        p0r = np.array([0, 1.3, 0.4, 0]).reshape((4, 1))
        pA = np.array([0, 1.495, 0.3, 0]).reshape((4, 1))
        pB = np.array([0, 1.495, 0.8, 0]).reshape((4, 1))
        pC = np.array([-1.0, 1.495, 0.8, 0]).reshape((4, 1))
        pD = np.array([-1.0, 1.495, 0.3, 0]).reshape((4, 1))

        rospy.loginfo("Gazebo's starting position: %s", str(self.theta_a))
        rospy.loginfo("tip_full starting position: %s", p0)

        self.Image_full = Image(
            [
                Hold(p0, 2.0),
                Goto(p0, p0r, 2.0),
                Goto(p0r, pA, 2.0),
                Hold(pA, 1.0),
                Goto(pA, pB, 2.0),
                Goto(pB, pC, 6.0),
                Goto(pC, pD, 2.0),
                Goto(pD, pA, 6.0),
                Goto(pA, p0r, 2.0),
                Hold(p0r, 1000.0),
            ]
        )

        T_2, _ = self.kin.fkin(self.theta_a[0:6], 1)
        p_2 = p_from_T(T_2).reshape(3)

        p0_2 = np.array([[p_2[0], p_2[1], p_2[2], -pi / 2]]).reshape((4, 1))
        p0r_2 = np.array([0.1, 0.6, 0.2, -pi / 2]).reshape((4, 1))
        pA_2 = np.array([0.295, 0.6, 0.2, -pi / 2]).reshape((4, 1))
        pB_2 = np.array([0.295, 0.6, 0.5, -pi / 2]).reshape((4, 1))
        pC_2 = np.array([0.295, 0.4, 0.5, -pi / 2]).reshape((4, 1))
        pD_2 = np.array([0.295, 0.4, 0.2, -pi / 2]).reshape((4, 1))
        rospy.loginfo("tip_elbow starting position: %s", p0_2)

        self.Image_elbow = Image(
            [
                Hold(p0_2, 2.0),
                Goto(p0_2, p0r_2, 2.0),
                Goto(p0r_2, pA_2, 2.0),
                Hold(pA_2, 1.0),
                Goto(pA_2, pB_2, 2.0),
                Goto(pB_2, pC_2, 2.0),
                Goto(pC_2, pD_2, 2.0),
                Goto(pD_2, pA_2, 2.0),
                Goto(pA_2, p0r_2, 2.0),
                Hold(p0r_2, 1000.0),
            ]
        )
        # Also reset the trajectory, starting at the beginning.
        self.reset()

        rospy.Subscriber("/twelvedof/joint_states",
                         JointState, self.jointCallback)

    def jointCallback(self, msg):
        self.theta_a = np.array(msg.position).reshape(12)
        # print("t", self.theta_a)

    # Reset.  If the simulation resets, also restart the trajectory.
    def reset(self):
        # Just reset the segment index counter and start time to zero.
        self.t0 = 0.0
        self.index = 0

    def calculate_canvas_attraction(self, tip_position):
        pde = np.array([0.0, 0.0, 0.0])
        for C in self.canvases:
            pc = p_from_T(C @ tip_position).reshape(3)
            if pc[1] < CANVAS_ATTRACTION_DISTANCE:
                pde += (
                    np.linalg.inv(R_from_T(C))
                    @ np.array(
                        [
                            0.0,
                            min(
                                0.0,
                                CANVAS_PRESSURE_SCALING *
                                (CANVAS_TARGET_DEPTH - pc[1]),
                            ),
                            0.0,
                        ]
                    )
                ).reshape(3)
        return pde

    # Update is called every 10ms of simulation time!
    def update(self, t, dt):
        # If the current trajectory segment is done, shift to the next.
        # Grab the spline output as joint values.
        (p_full, pd_full) = self.Image_full.update(t)
        (p_elbow, pd_elbow) = self.Image_elbow.update(t)

        # print("p", p)
        # print("pd", pd)

        # weights1 = np.eye(6)
        # np.fill_diagonal(weights1, [0.2, 2, 2, 5, 5, 5])

        full_pos, _ = self.kin.fkin(self.theta_a, 0)
        elbow_pos, _ = self.kin.fkin(self.theta_a[0:6], 1)

        full_canvas_attraction = self.calculate_canvas_attraction(full_pos)
        elbow_canvas_attraction = self.calculate_canvas_attraction(elbow_pos)

        (self.theta_d, _) = self.kin.velocity_ikin_combined(
            self.theta_d,
            self.theta_a,
            np.array([p_full[0], p_full[1], p_full[2]]).reshape((3, 1)),
            Rz(0),
            np.array([pd_full[0], pd_full[1], pd_full[2]]).reshape((3, 1)),
            np.array([0, 0, 0]).reshape((3, 1)),
            np.array([p_elbow[0], p_elbow[1], p_elbow[2]]).reshape((3, 1)),
            Rz(-pi / 2),
            np.array([pd_elbow[0], pd_elbow[1], pd_elbow[2]]).reshape((3, 1)),
            np.array([0, 0, 0]).reshape((3, 1)),
            1,
            dt,
        )
        # self.theta_d = np.append(theta_d1, theta_d2).reshape(12)

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
