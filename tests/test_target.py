import unittest

import pybullet as p
import pybullet_data
from rl_mus.agents.agents import Entity, Uav, UavCtrlType, Target
from rl_mus.utils.logger import UavLogger
import time
import numpy as np


class TestPlotter(unittest.TestCase):
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
            cameraYaw=-15,
            # cameraPitch=-30,
            cameraPitch=-45,
            cameraTargetPosition=[0, 0, 0],
            physicsClientId=self.client,
        )
        ret = p.getDebugVisualizerCamera(physicsClientId=self.client)
        print("viewMatrix", ret[2])
        print("projectionMatrix", ret[3])
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -9.81)
        self.planeId = p.loadURDF("plane.urdf")

    def tearDown(self) -> None:
        p.disconnect()

    def test_loading_target(self):
        start_pos = [0, 0, 1]

        target = Target(start_pos, client=self.client)

        # p.changeVisualShape(target.id, -1, rgbaColor=[0.5, 0.25])

        self.uav = Uav(
            [1,1,2], [0, 0, 0], client=self.client, ctrl_type=UavCtrlType.POS, show_local_axis=True
        )
        # p.changeVisualShape(self.uav.id, -1, rgbaColor=[0.8, 0.6, 0.4, 1])

        # plotter = Plotter(ctrl_type=UavCtrlType.POS)
        # plotter.add_uav(self.uav.id)
        pos_des = np.zeros(4)

        for i in range(10 * 240):
            pos_des[:3] = target.pos
            self.uav.step(pos_des)
            p.stepSimulation()
            # time.sleep(1 / 240)

            # plotter.log(self.uav.id, self.uav.state, pos_des)

        # plotter.plot("Test UAV reaching target", plt_ctrl=True)

if __name__ == "__main__":
    unittest.main()
