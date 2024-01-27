import unittest

import pybullet as p
import pybullet_data
from rl_mus_pybullet.agents.uav import Entity, Uav, UavCtrlType
from rl_mus_pybullet.utils.plot_utils import plot_traj
import time
import numpy as np


class TestUav(unittest.TestCase):
    def setUp(self) -> None:
        self.client = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        for i in [
            p.COV_ENABLE_RGB_BUFFER_PREVIEW,
            p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
            p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
        ]:
            p.configureDebugVisualizer(i, 0, physicsClientId=self.client)
        p.resetDebugVisualizerCamera(
            cameraDistance=3,
            cameraYaw=-30,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0],
            physicsClientId=self.client,
        )
        ret = p.getDebugVisualizerCamera(physicsClientId=self.client)
        print("viewMatrix", ret[2])
        print("projectionMatrix", ret[3])
        #### Add input sliders to the GUI ##########################
        # self.SLIDERS = -1*np.ones(4)
        # for i in range(4):
        # self.SLIDERS[i] = p.addUserDebugParameter("Propeller "+str(i)+" RPM", 0, self.MAX_RPM, self.HOVER_RPM, physicsClientId=self.client)
        # self.INPUT_SWITCH = p.addUserDebugParameter("Use GUI RPM", 9999, -1, 0, physicsClientId=self.client)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -9.81)
        self.planeId = p.loadURDF("plane.urdf")

    def tearDown(self) -> None:
        p.disconnect()

    @unittest.skip
    def test_init_uav(self):
        start_pos = [0, 0, 1]
        start_rpy = [0, 0, 0]
        self.uav = Uav(
            start_pos, start_rpy, client=self.client, ctrl_type=UavCtrlType.RPM
        )
        p.stepSimulation()
        print(self.uav.state)

    @unittest.skip
    def test_uav_hover_rpm(self):
        start_pos = [0, 0, 1]
        start_rpy = [0, 0, 0]
        self.uav = Uav(
            start_pos, start_rpy, client=self.client, ctrl_type=UavCtrlType.RPM
        )
        hover_rpms = self.uav.hover_rpm

        for i in range(240):
            p.stepSimulation()
            self.uav.step(hover_rpms)
            print(self.uav.state)

    @unittest.skip
    def test_uav_hover_vel(self):
        des_vel = np.zeros(3)
        self.uav = Uav(
            [1, 0, 1], [0, 0, 0], client=self.client, ctrl_type=UavCtrlType.VEL
        )

        des_pos = self.uav.state.copy()
        uav_des_traj = []
        uav_trajectory = []

        for i in range(10*240):
            p.stepSimulation()
            # time.sleep(1 / 240.0)
            self.uav.step(des_vel)

            uav_des_traj.append(des_pos.copy())
            uav_trajectory.append(self.uav.state.copy())

        uav_des_traj = np.array(uav_des_traj)
        uav_trajectory = np.array(uav_trajectory)

        plot_traj(uav_des_traj, uav_trajectory, title="Test Desired Controller")

    def test_uav_vel_tracking(self):
        self.uav = Uav(
            [3, 2, 1], [0, 0, 0], client=self.client, ctrl_type=UavCtrlType.VEL
        )
        des_pos = self.uav.state.copy()
        uav_des_traj = []
        uav_trajectory = []

        for i in range(40 * 240):
            des_pos = self.uav.state.copy()
            if i < 10 * 240:
                pass
            elif i >= 10 * 240 and i < 20 * 240:
                des_pos[10:13] = np.array([0, 0, 1])
            elif i >= 20 * 240:
                des_pos[10:13] = np.array([1, 0, 0])

            p.stepSimulation()
            self.uav.step(des_pos[10:13])
            uav_des_traj.append(des_pos.copy())
            uav_trajectory.append(self.uav.state.copy())

        uav_des_traj = np.array(uav_des_traj)
        uav_trajectory = np.array(uav_trajectory)

        plot_traj(uav_des_traj, uav_trajectory, title="Test Desired Controller")


if __name__ == "__main__":
    unittest.main()
