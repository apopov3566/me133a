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
from std_msgs.msg import Float64, Bool
from urdf_parser_py.urdf import Robot
from draw_controller import getCanvases

from math import pi, sin, cos

# Import the kinematics stuff:
from kinematics12 import Kinematics, p_from_T, R_from_T, Rx, Ry, Rz

# We could also import the whole thing ("import kinematics"),
# but then we'd have to write "kinematics.p_from_T()" ...

# Import the Spline stuff:
from splines import CubicSpline, Goto, Hold, Stay, QuinticSpline, Goto5, LinearSpline


CANVAS_ATTRACTION_DISTANCE = 0.07  # distance when pen starts pull to canvas
CANVAS_TARGET_DEPTH = 0  # target pull depth (stops pulling when reached)
CANVAS_PRESSURE_SCALING = 1.0  # strength of pull based on distance
CN_LIMIT = 50

#
#  Generator Class
#


class Image:
    def __init__(self, pos_fun, vel_fun, stop_u):
        self.pos_fun = pos_fun
        self.vel_fun = vel_fun
        self.stop_u = stop_u

    def evaluate(self, u):
        if (u > self.stop_u):
            return (self.pos_fun(self.stop_u), self.vel_fun(self.stop_u) * 0)
        return(self.pos_fun(u), self.vel_fun(u))


class ImageDriver:
    def __init__(self, image, acc, max_vel, pause_offset, pause_vel):
        self.image = image

        self.paused = False
        self.vel = 0
        self.pos = 0
        self.acc = acc
        self.max_vel = max_vel

        self.pause_offset = pause_offset
        self.pause_pos = 0
        self.pause_vel = pause_vel

    def pause(self):
        self.paused = True

    def unpause(self):
        self.paused = False

    def evaluate(self, dt, switch_vel):
        # adjust velocity
        if self.paused:
            self.vel = 0
        else:
            self.vel += dt * self.acc
            self.vel = min(self.vel, self.max_vel)

        # adjust position
        self.pos += self.vel * dt * switch_vel

        (im_pos, im_vel) = self.image.evaluate(self.pos)
        if self.paused:
            self.pause_pos += self.pause_vel * dt
            self.pause_pos = min(self.pause_pos, 1)

        else:
            self.pause_pos -= self.pause_vel * dt
            self.pause_pos = max(self.pause_pos, 0)

        # im_pos += self.pause_offset * self.pause_pos

        return (im_pos, self.vel * im_vel * switch_vel)


class ImagesDriver:
    def __init__(self, id_full, id_elbow, theta_init, conv_time, kin):
        self.id_full = id_full
        self.id_elbow = id_elbow
        self.kin = kin
        self.conv_time = conv_time

        p0_full = p_from_T(self.kin.fkin(theta_init, 0)[0]).reshape(3)
        p0_elbow = p_from_T(self.kin.fkin(theta_init[:6], 1)[0]).reshape(3)

        self.i_spline_full = Goto(
            p0_full, self.id_full.evaluate(0, 1)[0], conv_time, "Task")
        self.i_spline_elbow = Goto(
            p0_elbow, self.id_elbow.evaluate(0, 1)[0], conv_time, "Task")

    def pause_elbow(self):
        self.id_elbow.pause()

    def unpause_elbow(self):
        self.id_elbow.unpause()

    def evaluate(self, t, dt, switch_vel):
        if (t < self.conv_time + 1):
            if (t < self.conv_time):
                return(self.i_spline_full.evaluate(t), self.i_spline_elbow.evaluate(t))
            else:
                return(self.i_spline_full.evaluate(self.conv_time), self.i_spline_elbow.evaluate(self.conv_time))
        return (self.id_full.evaluate(dt, switch_vel), self.id_elbow.evaluate(dt, switch_vel))


class Generator:
    # Initialize.
    def __init__(self):
        # The Gazebo controllers treat each joint seperately.  We thus
        # need a seperate publisher for each joint under the topic
        # "/BOTNAME/CONTROLLER/command"...
        self.N = 12
        self.pubs = []

        self.canvases = getCanvases()

        self.extrudepub = rospy.Publisher("/extrude", Bool, queue_size=10)
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
        self.theta_a = self.ordered_joints_from_msg(msg)
        self.theta_d = self.ordered_joints_from_msg(msg)
        robot = Robot.from_parameter_server()
        self.kin = Kinematics(robot, "world", ["tip_full", "tip_elbow"])
        self.cn = self.kin.get_condition_number(self.theta_a)

        # IF we wanted to do kinematics:
        # # Grab the robot's URDF from the parameter server.

        # # Instantiate the Kinematics

        depth = 1.497

        spline_u = Goto(np.array([-0.1, depth, 0.5]),
                        np.array([-0.8, depth, 0.5]), 1.0)
        spline_l = Goto(np.array([-0.8, depth, 0.5]),
                        np.array([-0.8, depth, 0.6]), 0.5)
        spline_d = Goto(np.array([-0.8, depth, 0.6]),
                        np.array([-0.1, depth, 0.6]), 1.0)
        spline_r = Goto(np.array([-0.1, depth, 0.6]),
                        np.array([-0.1, depth, 0.5]), 0.5)

        def full_pos_fun(u):
            if u < 1.0:
                return spline_u.evaluate(u)[0]
            elif u < 1.5:
                return spline_l.evaluate(u - 1.0)[0]
            elif u < 2.5:
                return spline_d.evaluate(u - 1.5)[0]
            else:
                return spline_r.evaluate(u - 2.5)[0]

        def full_vel_fun(u):
            if u < 1.0:
                return spline_u.evaluate(u)[1]
            elif u < 1.5:
                return spline_l.evaluate(u - 1.0)[1]
            elif u < 2.5:
                return spline_d.evaluate(u - 1.5)[1]
            else:
                return spline_r.evaluate(u - 2.5)[1]

        spline1 = Goto(np.array([0.295, 0.6, 0.3]),
                       np.array([0.295, 0.6, 0.6]), 1.0)
        spline2 = Goto(np.array([0.295, 0.6, 0.6]),
                       np.array([0.295, 0.7, 0.6]), 0.5)
        spline3 = Goto(np.array([0.295, 0.7, 0.6]),
                       np.array([0.295, 0.7, 0.3]), 1.0)
        spline4 = Goto(np.array([0.295, 0.7, 0.3]),
                       np.array([0.295, 0.6, 0.3]), 0.5)

        def elbow_pos_fun(u):
            if u < 1.0:
                return spline1.evaluate(u)[0]
            elif u < 1.5:
                return spline2.evaluate(u - 1.0)[0]
            elif u < 2.5:
                return spline3.evaluate(u - 1.5)[0]
            else:
                return spline4.evaluate(u - 2.5)[0]

        def elbow_vel_fun(u):
            if u < 1.0:
                return spline1.evaluate(u)[1]
            elif u < 1.5:
                return spline2.evaluate(u - 1.0)[1]
            elif u < 2.5:
                return spline3.evaluate(u - 1.5)[1]
            else:
                return spline4.evaluate(u - 2.5)[1]

        self.ID = ImagesDriver(ImageDriver(Image(full_pos_fun, full_vel_fun, 3.0), 1, 0.1, np.array([0, 0, 0]), 10),
                               ImageDriver(
                                   Image(elbow_pos_fun, elbow_vel_fun, 3.0), 1, 0.1, np.array([0, 0, 0]), 10),
                               self.theta_a, 10, self.kin)

        self.pause_time = None
        self.wait_time = 0.1

        rospy.Subscriber("/twelvedof/joint_states",
                         JointState, self.jointCallback)

    def ordered_joints_from_msg(self, msg):
        p = np.array(msg.position).reshape(12)
        return np.array([p[0], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[1], p[2], p[3]])

    def jointCallback(self, msg):
        # print(msg.)
        self.theta_a = self.ordered_joints_from_msg(msg)
        self.cn = self.kin.get_condition_number(self.theta_a)
        print(self.cn)
        # print("t", self.theta_a)

    def track_elbow(self, p_elbow):
        p_elbow_a = p_from_T(self.kin.fkin(self.theta_a[:6], 1)[0])
        d = np.linalg.norm(p_elbow.reshape(3) - p_elbow_a.reshape(3))
        return self.cn < CN_LIMIT and d < 0.01

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
        switch_vel = max(
            0.2, min(1, (abs(CN_LIMIT - self.cn) * 0.05) ** (1/2)))

        ((p_full, pd_full), (p_elbow, pd_elbow)
         ) = self.ID.evaluate(t, dt, switch_vel)
        track = self.track_elbow(p_elbow)

        # print(switch_vel, self.cn, track)

        if not track:
            self.ID.pause_elbow()
            self.pause_time = t
        elif self.pause_time is None or t > self.pause_time + self.wait_time:
            self.ID.unpause_elbow()
            self.pause_time = None

        (self.theta_d, _, keep_num) = self.kin.velocity_ikin_combined(
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
            4 if track else 3
        )

        # Send the individal angle commands.
        for i in range(self.N):
            self.pubs[i].publish(Float64(self.theta_d[i]))

        self.extrudepub.publish(track)


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
