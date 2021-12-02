#!/usr/bin/env python3
#
#   kinematics.py
#
#   TO IMPORT, ADD TO YOUR CODE:
#   from kinematics import Kinematics, p_from_T, R_from_T, Rx, Ry, Rz
#
#
#   Kinematics Class and Helper Functions
#
#   This computes the forward kinematics and Jacobian using the
#   kinematic chain.  It also includes test code when run
#   independently.
#
import rospy
import numpy as np

from urdf_parser_py.urdf import Robot


#
#  Kinematics Helper Functions
#
#  These helper functions simply convert the information between
#  different formats.  For example, between a python list and a NumPy
#  array.  Or between Euler Angles and a Rotation Matrix.  And so
#  forth.  They operate on:
#
#    NumPy 3x1  "p" Point vectors
#    NumPy 3x1  "e" Axes of rotation
#    NumPy 3x3  "R" Rotation matrices
#    NumPy 1x4  "q" Quaternions
#    NumPy 4x4  "T" Transforms
#
#  and may take inputs from the URDF tags <origin> and <axis>:
#
#    Python List 1x3:  <axis>          information
#    Python List 1x6:  <origin>        information
#    Python List 1x3:  <origin> "xyz"  vector of positions
#    Python List 1x3:  <origin> "rpy"  vector of angles
#

# Build T matrix from R/p.  Extract R/p from T matrix

# Joint order
#   - theta1
#   - theta10
#   - theta11
#   - theta12
#   - theta2
#   - theta3
#   - theta4
#   - theta5
#   - theta6
#   - theta7
#   - theta8
#   - theta9


def T_from_Rp(R, p):
    return np.vstack((np.hstack((R, p)), np.array([0.0, 0.0, 0.0, 1.0])))


def p_from_T(T):
    return T[0:3, 3:4]


def R_from_T(T):
    return T[0:3, 0:3]


# Basic Rotation Matrices about an axis: Rotx/Roty/Rotz/Rot(axis)


def Rx(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def Ry(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def Rz(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def R_from_axisangle(axis, theta):
    ex = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + np.sin(theta) * ex + (1.0 - np.cos(theta)) * ex @ ex


# Quaternion To/From Rotation Matrix


def R_from_q(q):
    norm2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]
    return -np.eye(3) + (2 / norm2) * (
        np.array(
            [
                [
                    q[1] * q[1] + q[0] * q[0],
                    q[1] * q[2] - q[0] * q[3],
                    q[1] * q[3] + q[0] * q[2],
                ],
                [
                    q[2] * q[1] + q[0] * q[3],
                    q[2] * q[2] + q[0] * q[0],
                    q[2] * q[3] - q[0] * q[1],
                ],
                [
                    q[3] * q[1] - q[0] * q[2],
                    q[3] * q[2] + q[0] * q[1],
                    q[3] * q[3] + q[0] * q[0],
                ],
            ]
        )
    )


def q_from_R(R):
    A = [
        1.0 + R[0][0] + R[1][1] + R[2][2],
        1.0 + R[0][0] - R[1][1] - R[2][2],
        1.0 - R[0][0] + R[1][1] - R[2][2],
        1.0 - R[0][0] - R[1][1] + R[2][2],
    ]
    i = A.index(max(A))
    A = A[i]
    c = 0.5 / np.sqrt(A)
    if i == 0:
        q = c * np.array([A, R[2][1] - R[1][2], R[0]
                         [2] - R[2][0], R[1][0] - R[0][1]])
    elif i == 1:
        q = c * np.array([R[2][1] - R[1][2], A, R[1]
                         [0] + R[0][1], R[0][2] + R[2][0]])
    elif i == 2:
        q = c * np.array([R[0][2] - R[2][0], R[1][0] +
                         R[0][1], A, R[2][1] + R[1][2]])
    else:
        q = c * np.array([R[1][0] - R[0][1], R[0][2] +
                         R[2][0], R[2][1] + R[1][2], A])
    return q


def q_from_T(T):
    return q_from_R(R_from_T(T))


# From URDF <ORIGIN> elements (and "xyz"/"rpy" sub-elements):
def T_from_URDF_origin(origin):
    return T_from_Rp(R_from_URDF_rpy(origin.rpy), p_from_URDF_xyz(origin.xyz))


def R_from_URDF_rpy(rpy):
    return Rz(rpy[2]) @ Ry(rpy[1]) @ Rx(rpy[0])


def p_from_URDF_xyz(xyz):
    return np.array(xyz).reshape((3, 1))


# From URDF <AXIS> elements:
def T_from_URDF_axisangle(axis, theta):
    return T_from_Rp(R_from_axisangle(axis, theta), np.zeros((3, 1)))


def e_from_URDF_axis(axis):
    return np.array(axis).reshape((3, 1))


#
#   Kinematics Class
#
#   This encapsulates the kinematics functionality, storing the
#   kinematic chain elements.
#
class Kinematics:
    def __init__(self, robot, baseframe, tipframes):
        # Report what we are doing.

        self.dofs = []
        self.joints = []

        for tipframe in tipframes:
            rospy.loginfo(
                "Kinematics: Setting up the chain from '%s' to '%s'...",
                baseframe,
                tipframe,
            )

            # Create the list of joints from the base frame to the tip
            # frame.  Search backwards, as this could be a tree structure.
            # Meantine while a parent may have multiple children, every
            # child has only one parent!  That makes the chain unique.
            self.joints.append([])
            frame = tipframe
            while frame != baseframe:
                joint = next(
                    (j for j in robot.joints if j.child == frame), None)
                if joint is None:
                    rospy.logerr("Unable find joint connecting to '%s'", frame)
                    raise Exception()
                if joint.parent == frame:
                    rospy.logerr(
                        "Joint '%s' connects '%s' to itself", joint.name, frame
                    )
                    raise Exception()
                self.joints[-1].insert(0, joint)
                frame = joint.parent

            # Report we found.
            self.dofs.append(
                sum(1 for j in self.joints[-1] if j.type != "fixed"))
            rospy.loginfo(
                "Kinematics: %d active DOFs, %d total steps",
                self.dofs[-1],
                len(self.joints[-1]),
            )

    def pe(self, pd, p):
        return pd - p

    def Re(self, Rd, R):
        s = [0, 0, 0]
        for i in range(3):
            s += np.cross(R[0:3, i].reshape((1, 3)),
                          Rd[0:3, i].reshape((1, 3)))
        return 1 / 2 * s.reshape(3, 1)

    def velocity_ikin_keep_axes(
        self,
        J_combined,
        vd_combined,
        pe_combined,
        theta_a,
        theta_target,
        l,
        keep_axes
    ):
        joint_mins = {}
        joint_maxs = {8: 0.8}
        g_max_min = 1
        theta_extra = np.array([0.0] * self.dofs[0])

        for joint in joint_mins:
            theta_extra[joint] += g_max_min * \
                max(0, (joint_mins[joint] -
                        theta_a.reshape(self.dofs[0])[joint]))
        for joint in joint_maxs:
            theta_extra[joint] += g_max_min * \
                min(0, (joint_maxs[joint] -
                        theta_a.reshape(self.dofs[0])[joint]))

        g_track = 10
        theta_extra[:6] += g_track * np.dot(
            np.linalg.inv(J_combined[6:, :6]),
            vd_combined[6:]
            + l * pe_combined[6:],
        ).reshape(6)

        # g_center = 0.0
        # for joint in range(self.dofs[0]):
        #     theta_extra[joint] += g_center * \
        #         (0.0 - theta_a.reshape(self.dofs[0])[joint])

        J_modified = J_combined[keep_axes.flatten(), :]
        return (
            np.dot(
                np.linalg.pinv(J_modified),
                vd_combined[keep_axes.flatten()]
                + l
                * pe_combined[keep_axes.flatten()],
            ) + np.dot(np.eye(self.dofs[0]) - (np.linalg.pinv(J_modified) @ J_modified), theta_extra.reshape(self.dofs[0], 1))
        ).reshape(self.dofs[0])

    def get_condition_number(self, theta_a):
        _, J_full_a = self.fkin(theta_a, 0)
        _, J_elbow_a = self.fkin(theta_a[:6], 1)
        J_combined = np.vstack(
            (J_full_a, np.hstack((J_elbow_a, np.zeros((6, 6))))))
        return np.linalg.cond(J_combined)

    def velocity_ikin_combined(
        self,
        theta_d,
        theta_a,
        pd_full,
        Rd_full,
        vd_full,
        wd_full,
        pd_elbow,
        Rd_elbow,
        vd_elbow,
        wd_elbow,
        l,
        dt,
        keep_num
    ):
        T_full, J_full = self.fkin(theta_d, 0)
        T_elbow, J_elbow = self.fkin(theta_d[:6], 1)

        T_full_a, J_full_a = self.fkin(theta_a, 0)
        T_elbow_a, J_elbow_a = self.fkin(theta_a[:6], 1)

        p_full = p_from_T(T_full)
        p_full_a = p_from_T(T_full_a)
        R_full = R_from_T(T_full)
        R_full_a = R_from_T(T_full_a)

        p_elbow = p_from_T(T_elbow)
        p_elbow_a = p_from_T(T_elbow_a)
        R_elbow = R_from_T(T_elbow)
        R_elbow_a = R_from_T(T_elbow_a)

        J_combined = np.vstack(
            (J_full_a, np.hstack((J_elbow_a, np.zeros((6, 6))))))

        keep_order = np.array([[0, 1, 2], [3, 4, 5], [9, 10, 11], [8, 7, 6]])

        theta_vel = self.velocity_ikin_keep_axes(
            J_combined,
            np.vstack((vd_full, wd_full, vd_elbow, wd_elbow)),
            np.vstack((
                self.pe(pd_full, p_full_a),
                self.Re(Rd_full, R_full_a),
                self.pe(pd_elbow, p_elbow_a),
                self.Re(Rd_elbow, R_elbow_a),
            )),
            theta_a,
            {},
            l,
            keep_order[0:keep_num]
        )

        theta_new = theta_d + dt * theta_vel

        return (theta_new, theta_vel, keep_num)

    def fkin(self, theta, chain=0):
        # Check the number of joints
        if len(theta) != self.dofs[chain]:
            rospy.logerr(
                "Number of joint angles (%d) does not match URDF (%d)",
                len(theta),
                self.dofs[chain],
            )
            return

        # Initialize the T matrix to walk up the chain
        T = np.eye(4)

        # As we are walking up, also store the position and joint
        # axis, ONLY FOR ACTIVE/MOVING/"REAL" joints.  We simply put
        # each in a python, and keep an index counter.
        plist = []
        elist = []
        index = 0

        # Walk the chain, one URDF <joint> entry at a time.  Each can
        # be "fixed" (just a transform) or "continuous" (a transform
        # AND a rotation).  NOTE the URDF entries are only the
        # step-by-step transformations.  That is, the information is
        # *not* in world frame - we have to append to the chain...
        for joint in self.joints[chain]:
            if joint.type == "fixed" or joint.type == "prismatic":
                # Just append the fixed transform
                T = T @ T_from_URDF_origin(joint.origin)

            elif joint.type == "continuous":
                # First append the fixed transform, then rotating
                # transform.  The joint angle comes from theta-vector.
                T = T @ T_from_URDF_origin(joint.origin)
                T = T @ T_from_URDF_axisangle(joint.axis, theta[index])

                # Save the position
                plist.append(p_from_T(T))

                # Save the joint axis.  The URDF <axis> is given in
                # the local frame, so multiply by the local R matrix.
                elist.append(R_from_T(T) @ e_from_URDF_axis(joint.axis))

                # Advanced the "active/moving" joint number
                index += 1

            elif joint.type != "fixed":
                # There shouldn't be any other types...
                rospy.logwarn("Unknown Joint Type: %s", joint.type)

        # Compute the Jacobian.  For that we need the tip information,
        # which is where the kinematic chain ended up, i.e. at T.
        ptip = p_from_T(T)
        J = np.zeros((6, index))
        for i in range(index):
            J[0:3, i: i + 1] = np.cross(elist[i], ptip - plist[i], axis=0)
            J[3:6, i: i + 1] = elist[i]

        # Return the Ttip and Jacobian (at the end of the chain).
        return (T, J)


#
#  Main Code (only if run independently):
#
#  Test assuming a 3DOF URDF is loaded.
#
if __name__ == "__main__":
    # Prepare/initialize this node.
    rospy.init_node("kinematics")

    # Grab the robot's URDF from the parameter server.
    robot = Robot.from_parameter_server()

    # Instantiate the Kinematics.  Generally we care about tip w.r.t. world!
    kin = Kinematics(robot, "world", "tip")

    # Pick the test angles of the robot.
    theta = [np.pi / 4, np.pi / 6, np.pi / 3]

    # Compute the kinematics.
    (T, J) = kin.fkin(theta)

    # Report.
    np.set_printoptions(precision=6, suppress=True)
    print("T:\n", T)
    print("J:\n", J)
    print("p/q: ", p_from_T(T).T, q_from_R(R_from_T(T)))
