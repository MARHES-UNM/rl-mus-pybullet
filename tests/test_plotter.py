import unittest

import pybullet as p
import pybullet_data
from rl_mus.agents.agents import Entity, Uav, UavCtrlType
from rl_mus.utils.plot_utils import Plotter
import time
import numpy as np


class TestPlotter(unittest.TestCase):
    def setUp(self) -> None:
        self.client = p.connect(p.DIRECT)  # or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -9.81)
        self.planeId = p.loadURDF("plane.urdf")

    def tearDown(self) -> None:
        p.disconnect()

    # @unittest.skip
    def test_plotter_1_uav(self):
        plotter = Plotter(num_uavs=1)

        start_pos = [0, 0, 1]
        start_rpy = [0, 0, 0]

        self.uav = Uav(
            start_pos, start_rpy, client=self.client, ctrl_type=UavCtrlType.VEL
        )

        plotter.add_uav(self.uav.id)

        max_vel = 0.5 * self.uav.vel_lim
        for i in range(2 * 240):
            ref_ctrl = np.random.uniform(low=-max_vel, high=max_vel, size=(3,))
            self.uav.step(ref_ctrl)
            p.stepSimulation()
            plotter.log(uav_id=self.uav.id, ref_ctrl=ref_ctrl, state=self.uav.state)

        plotter.plot(plt_ctrl=True)

    def test_plotter_multi_uav(self):
        plotter = Plotter(num_uavs=2)

        uav_1 = Uav([0, 0, 1], [0, 0, 0], client=self.client, ctrl_type=UavCtrlType.VEL)

        uav_2 = Uav([1, 0, 1], [0, 0, 0], client=self.client, ctrl_type=UavCtrlType.VEL)

        plotter.add_uav(uav_1.id)
        plotter.add_uav(uav_2.id)

        for i in range(2 * 240):
            ref_ctrl1 = np.random.uniform(
                low=-uav_1.vel_lim, high=uav_2.vel_lim, size=(3,)
            )
            ref_ctrl2 = np.random.uniform(
                low=-uav_1.vel_lim, high=uav_2.vel_lim, size=(3,)
            )

            uav_1.step(ref_ctrl1)
            uav_2.step(ref_ctrl2)

            p.stepSimulation()
            plotter.log(uav_id=uav_1.id, ref_ctrl=ref_ctrl1, state=uav_1.state)
            plotter.log(uav_id=uav_2.id, ref_ctrl=ref_ctrl2, state=uav_2.state)

        plotter.plot()


if __name__ == "__main__":
    unittest.main()
