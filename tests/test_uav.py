import unittest

import pybullet as p
import pybullet_data
from rl_mus_pybullet.agents.uav import Uav
import time


class TestUav(unittest.TestCase):
    def setUp(self) -> None:
        self.client = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -9.81)
        self.planeId = p.loadURDF("plane.urdf")

    def tearDown(self) -> None:
        p.disconnect()

    @unittest.skip
    def test_init_uav(self):
        start_pos = [0, 0, 1]
        start_rpy = [0, 0, 0]
        uav = Uav(start_pos, start_rpy, client=self.client)
        p.stepSimulation()
        print(uav.state)

    def test_uav_hover(self):
        start_pos = [0, 0, .5]
        start_rpy = [0, 0, 0]
        uav = Uav(start_pos, start_rpy, client=self.client)
        for i in range(10000):
            p.stepSimulation()
            uav.step()
            time.sleep(1/240)
            print(uav.state)


if __name__ == "__main__":
    unittest.main()
