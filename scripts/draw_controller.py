#!/usr/bin/env python3
import rospy
import numpy as np

from gazebo_msgs.msg import LinkStates, ModelStates
from sensor_msgs.msg import JointState
from project.msg import DrawingTip
from urdf_parser_py.urdf import Robot

from kinematics import T_from_URDF_origin, T_from_Rp, R_from_q, p_from_T


CANVAS_WIDTH = 1
CANVAS_HEIGHT = 1
CANVAS_DEPTH = 0.01
MIN_DRAW_DISTANCE = 0.003


def getCanvases():
    canvases = []
    msg = rospy.wait_for_message("/gazebo/model_states", ModelStates)
    for i in range(len(msg.name)):
        if(msg.name[i].startswith("canvas")):
            canvases.append(np.linalg.inv(T_from_Rp(R_from_q([msg.pose[i].orientation.w, msg.pose[i].orientation.x, msg.pose[i].orientation.y, msg.pose[i].orientation.z]), np.array(
                [msg.pose[i].position.x, msg.pose[i].position.y, msg.pose[i].position.z]).reshape(3, 1))))
    return canvases


class Controller():
    def __init__(self):
        self.tippub = rospy.Publisher('/tipdraw', DrawingTip, queue_size=10)
        self.T = self.getLinkToTipTransform("link7", "tip")
        rospy.loginfo(self.T)

        self.canvases = getCanvases()

        self.last_marker = None

        self.listener()

    def getLinkToTipTransform(self, link, tip):
        robot = Robot.from_parameter_server()

        frame = tip
        joints = []

        while (frame != link):
            joint = next((j for j in robot.joints if j.child == frame), None)
            if (joint is None):
                rospy.logerr("Unable find joint connecting to '%s'", frame)
                raise Exception()
            if (joint.parent == frame):
                rospy.logerr("Joint '%s' connects '%s' to itself",
                             joint.name, frame)
                raise Exception()
            if (joint.type != 'fixed'):
                rospy.logerr("non-fixed joint between final link and tip")
                raise Exception()
            joints.insert(0, joint)
            frame = joint.parent

        T = np.eye(4)
        for joint in joints:
            T = T @ T_from_URDF_origin(joint.origin)

        return T

    def callback(self, data):
        i = data.name.index("sevendof::link7")

        L = T_from_Rp(R_from_q([data.pose[i].orientation.w, data.pose[i].orientation.x, data.pose[i].orientation.y, data.pose[i].orientation.z]), np.array(
            [data.pose[i].position.x, data.pose[i].position.y, data.pose[i].position.z]).reshape(3, 1))
        P = L @ self.T
        p = p_from_T(P).reshape(3)

        touch = False
        for C in self.canvases:
            pc = p_from_T(C @ P).reshape(3)
            if pc[1] < CANVAS_DEPTH and abs(pc[0]) < CANVAS_WIDTH and abs(pc[2]) < CANVAS_HEIGHT:
                touch = True
                break

        if touch and (self.last_marker is None or np.linalg.norm(self.last_marker - p) > MIN_DRAW_DISTANCE):
            self.last_marker = p
            drawingtip = DrawingTip()
            drawingtip.x = p[0]
            drawingtip.y = p[1]
            drawingtip.z = p[2]
            self.tippub.publish(drawingtip)

    def listener(self):
        rospy.Subscriber("/gazebo/link_states", LinkStates, self.callback)

        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('draw_controller')
    c = Controller()
