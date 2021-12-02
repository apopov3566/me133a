#!/usr/bin/env python3
import rospy
import numpy as np

from gazebo_msgs.msg import LinkStates, ModelStates
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from project.msg import DrawingTip
from urdf_parser_py.urdf import Robot

from math import floor

from kinematics import T_from_URDF_origin, T_from_Rp, R_from_q, p_from_T


CANVAS_WIDTH = 1
CANVAS_HEIGHT = 1
CANVAS_DEPTH = 0.01
MIN_DRAW_DISTANCE = 0.01


def getCanvases():
    canvases = []
    msg = rospy.wait_for_message("/gazebo/model_states", ModelStates)
    for i in range(len(msg.name)):
        if msg.name[i].startswith("canvas"):
            canvases.append(
                np.linalg.inv(
                    T_from_Rp(
                        R_from_q(
                            [
                                msg.pose[i].orientation.w,
                                msg.pose[i].orientation.x,
                                msg.pose[i].orientation.y,
                                msg.pose[i].orientation.z,
                            ]
                        ),
                        np.array(
                            [
                                msg.pose[i].position.x,
                                msg.pose[i].position.y,
                                msg.pose[i].position.z,
                            ]
                        ).reshape(3, 1),
                    )
                )
            )
    return canvases


class Controller:
    def __init__(self):
        self.tippub = rospy.Publisher("/tipdraw", DrawingTip, queue_size=10)
        self.Ts = {
            "link6": self.getLinkToTipTransform("link6", "tip_elbow"),
            "link12": self.getLinkToTipTransform("link12", "tip_full"),
        }

        self.extrude = False

        rospy.loginfo(self.Ts)

        self.canvases = getCanvases()

        self.marks = []
        for _ in self.canvases:
            self.marks.append(np.zeros((200, 200)))

        self.last_marker = None

        self.listener()

    def getLinkToTipTransform(self, link, tip):
        robot = Robot.from_parameter_server()

        frame = tip
        joints = []

        while frame != link:
            joint = next((j for j in robot.joints if j.child == frame), None)
            if joint is None:
                rospy.logerr("Unable find joint connecting to '%s'", frame)
                raise Exception()
            if joint.parent == frame:
                rospy.logerr("Joint '%s' connects '%s' to itself",
                             joint.name, frame)
                raise Exception()
            if joint.type != "fixed":
                rospy.logerr("non-fixed joint between final link and tip")
                raise Exception()
            joints.insert(0, joint)
            frame = joint.parent

        T = np.eye(4)
        for joint in joints:
            T = T @ T_from_URDF_origin(joint.origin)

        return T

    def extrudecallback(self, data):
        self.extrude = data.data

    def callback(self, data):
        for link in self.Ts:
            T = self.Ts[link]

            i = data.name.index("twelvedof::" + link)

            L = T_from_Rp(
                R_from_q(
                    [
                        data.pose[i].orientation.w,
                        data.pose[i].orientation.x,
                        data.pose[i].orientation.y,
                        data.pose[i].orientation.z,
                    ]
                ),
                np.array(
                    [
                        data.pose[i].position.x,
                        data.pose[i].position.y,
                        data.pose[i].position.z,
                    ]
                ).reshape(3, 1),
            )
            P = L @ T
            p = p_from_T(P).reshape(3)

            for C, m in zip(self.canvases, self.marks):
                pc = p_from_T(C @ P).reshape(3)

                m_height, m_width = m.shape
                w = floor((pc[0] + CANVAS_WIDTH) / 2 / CANVAS_WIDTH * m_width)
                h = floor((pc[2] + CANVAS_HEIGHT) /
                          2 / CANVAS_HEIGHT * m_height)

                if (
                    pc[1] < CANVAS_DEPTH
                    and w > 0 and w < m_width
                    and h > 0 and h < m_height
                    and m[h, w] == 0
                    and (link == "link12" or self.extrude)
                ):
                    m[h, w] = 1

                    ps = np.array([pc[0], 0, pc[2], 1]).reshape((4, 1))
                    surface_mark = (np.linalg.inv(C) @ ps).reshape(4)
                    drawingtip = DrawingTip()
                    drawingtip.x = surface_mark[0]
                    drawingtip.y = surface_mark[1]
                    drawingtip.z = surface_mark[2]
                    self.tippub.publish(drawingtip)
                    break

    def listener(self):
        rospy.Subscriber("/gazebo/link_states", LinkStates, self.callback)
        rospy.Subscriber("/extrude", Bool, self.extrudecallback)

        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node("draw_controller")
    c = Controller()
