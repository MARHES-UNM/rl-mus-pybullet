import unittest

import pybullet as p
import pybullet_data
from rl_mus_pybullet.agents.uav import Entity, Uav
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

    # @unittest.skip
    def test_init_uav(self):
        start_pos = [0, 0, 1]
        start_rpy = [0, 0, 0]
        x = Entity(start_pos, start_rpy, client=self.client)
        for i in range(240):
            p.stepSimulation()
            time.sleep(1 / 240.0)

    def test_uav_hover(self):
        start_pos = [0, 0, 0.5]
        start_rpy = [0, 0, 0]
        uav = Uav(start_pos, start_rpy, client=self.client)
        kf = 3.16e-10
        km = 7.94e-12
        g = 9.81
        m = 0.027
        rpms = np.array([np.sqrt(g * m / (4 * kf))] * 4)

        for i in range(10000):
            p.stepSimulation()
            uav.step(rpms)
            time.sleep(1 / 240)
            print(uav.state)


if __name__ == "__main__":
    unittest.main()
