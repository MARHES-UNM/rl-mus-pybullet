import unittest

import pybullet as p
import pybullet_data
from rl_mus.agents.agents import Entity, Uav, UavCtrlConf, UavCtrlType
from rl_mus.utils.logger import UavLogger, plot_traj
import time
import numpy as np
from rl_mus.utils.math_utils import wrap_angle


class TestUav(unittest.TestCase):
    def setUp(self) -> None:
        use_gui = False
        if use_gui:
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

            # # ### Add input sliders to the GUI ##########################
            # # self.SLIDERS = -1*np.ones(4)
            # # for i in range(4):
            # # self.SLIDERS[i] = p.addUserDebugParameter("Propeller "+str(i)+" RPM", 0, self.MAX_RPM, self.HOVER_RPM, physicsClientId=self.client)
            # # self.INPUT_SWITCH = p.addUserDebugParameter("Use GUI RPM", 9999, -1, 0, physicsClientId=self.client)

        else:
            self.client = p.connect(p.DIRECT)  # or p.DIRECT for non-graphical version

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -9.81)
        self.planeId = p.loadURDF("plane.urdf")
        self.n_secs_short = 2
        self.n_secs_med = 4
        self.n_secs_long = 10

    def tearDown(self) -> None:
        p.disconnect()

    def test_init_uav(self):
        start_pos = [0, 0, 1]
        start_rpy = [0, 0, 0]
        self.uav = Uav(
            start_pos,
            start_rpy,
            client=self.client,
            ctrl_type=UavCtrlType.RPM,
            ctrl_conf=UavCtrlConf.P,
        )
        p.stepSimulation()
        print(self.uav.state)

    def test_uav_hover_rpm(self):
        start_pos = [0, 0, 1]
        start_rpy = [0, 0, 0]
        self.uav = Uav(
            start_pos, start_rpy, client=self.client, ctrl_type=UavCtrlType.RPM
        )

        plotter = UavLogger(ctrl_type=UavCtrlType.RPM)
        plotter.add_uav(self.uav.id)

        rpms_des = np.zeros(4)

        n_steps = int(self.n_secs_short * self.uav.pyb_freq)
        for i in range(n_steps):
            self.uav.step(rpms_des)
            p.stepSimulation()

            plotter.log(self.uav.id, state=self.uav.state, action=rpms_des)

        plotter.plot(plt_action=True, title="Test Desired RPMs to Hover.")

    # @unittest.skip
    def test_uav_hover_vel(self):
        self.uav = Uav(
            [1, 0, 1], [0, 0, 0], client=self.client, ctrl_type=UavCtrlType.VEL
        )

        plotter = UavLogger(ctrl_type=UavCtrlType.VEL)
        plotter.add_uav(self.uav.id)

        vel_des = np.zeros(4)
        n_steps = int(self.n_secs_short * self.uav.pyb_freq)

        for i in range(n_steps):
            self.uav.step(vel_des)
            p.stepSimulation()

            plotter.log(self.uav.id, self.uav.state, vel_des)

        plotter.plot("Test UAV Hover Velocity", plt_action=True)

    def test_uav_vel_tracking(self):
        self.uav = Uav(
            [3, 2, 1], [0, 0, 0], client=self.client, ctrl_type=UavCtrlType.VEL
        )

        plotter = UavLogger()
        plotter.add_uav(self.uav.id)

        vel_des = np.zeros(4)
        vel_des[3] = self.uav.vel_lim

        n_steps = int(self.n_secs_long * self.uav.pyb_freq)
        for i in range(n_steps):
            if i < 1 * 240:
                pass
            elif i >= 1 * 240 and i < 2 * 240:
                vel_des[:3] = np.array([0, 0, 1.0])

                print(vel_des)
            elif i >= 2 * 240 and i < 3 * 240:
                vel_des[:3] = np.array([0, 1.0, 0])

            elif i >= 3 * 240 and i < 4 * 240:
                vel_des[:3] = np.array([1.0, 0, 0])

            elif i >= 4 * 240 and i < 5 * 240:
                vel_des[:3] = np.array([1.0, 1.0, 1.0])

            elif i >= 6 * 240 and i < 7 * 240:
                vel_des[:3] = np.array([-1.0, 0.0, 0.0])

            elif i >= 7 * 240 and i < 8 * 240:
                vel_des[:3] = np.array([0.0, -0.5, 0.0])

            elif i >= 8 * 240 and i < 9 * 240:
                vel_des[:3] = np.array([0.0, 0.0, -0.5])

            elif i >= 9 * 240:
                vel_des[:3] = np.array([0, 0, 0])

            self.uav.step(vel_des)
            p.stepSimulation()

            plotter.log(self.uav.id, state=self.uav.state.copy(), action=vel_des.copy())

        plotter.plot(title="Test UAV velocity control", plt_action=True)

    def test_uav_fast_vel_tracking(self):
        self.uav = Uav(
            [1, 1, 1], [0, 0, 0], client=self.client, ctrl_type=UavCtrlType.VEL
        )

        plotter = UavLogger()
        plotter.add_uav(self.uav.id)

        time_to_change_vel = .1 * 240  # every 2 secs
        vel_des = np.zeros(4)
        max_vel = self.uav.vel_lim * 0.5
        for i in range(10 * 240):
            if i % time_to_change_vel == 0:
                vel_des[:3] = np.random.uniform(low=-max_vel, high=max_vel, size=(3,))
                vel_des[3] = max_vel

            self.uav.step(vel_des)
            p.stepSimulation()

            plotter.log(uav_id=self.uav.id, state=self.uav.state.copy(), action=vel_des.copy())

        plotter.plot(title="Test Fast velocity tracking", plt_action=True)

    # @unittest.skip
    def test_uav_rand_vel_tracking(self):
        self.uav = Uav(
            [0, 0, 0.5], [0, 0, 0], client=self.client, ctrl_type=UavCtrlType.VEL
        )

        plotter = UavLogger()
        plotter.add_uav(self.uav.id)

        time_to_change_vel = int(240 / 0.5) # every 2 secs
        vel_des = np.zeros(4)
        max_vel = self.uav.vel_lim * 0.25
        for i in range(10 * 240):
            if i % time_to_change_vel == 0:
                vel_des[:3] = np.random.uniform(low=-max_vel, high=max_vel, size=(3,))
                vel_des[3] = max_vel 

            self.uav.step(vel_des)
            p.stepSimulation()
            plotter.log(uav_id=self.uav.id, state=self.uav.state.copy(), action=vel_des.copy())

        plotter.plot(title="Test Random Velocity Tracking", plt_action=True)

    # @unittest.skip
    def test_uav_circular_traj_pos(self):
        """
        https://pressbooks.online.ucf.edu/phy2048tjb/chapter/4-4-uniform-circular-motion/
        """
        self.uav = Uav(
            [0, 0, 2], [0, 0, 0], client=self.client, ctrl_type=UavCtrlType.POS
        )

        plotter = UavLogger(ctrl_type=UavCtrlType.POS)
        plotter.add_uav(self.uav.id)
        action = np.zeros(4)

        # this determines how fast to complete a circle
        circ_freq = 1.0 / (240.0 * 2.0)  # hz
        circ_rad = 0.5
        for i in range(10 * 240):
            action[0] = circ_rad * np.cos(2.0 * np.pi * circ_freq * i)
            action[1] = circ_rad * np.sin(2.0 * np.pi * circ_freq * i)
            action[2] = 0.9
            action[3] = wrap_angle(180 * i * np.pi / 180 * circ_freq)

            self.uav.step(action)
            p.stepSimulation()
            plotter.log(
                uav_id=self.uav.id, state=self.uav.state.copy(), action=action.copy()
            )

        plotter.plot(title="Test Circular Velocity Tracking", plt_action=True)

    def test_uav_helix_vel_tracking(self):
        self.uav = Uav(
            [0, 0, 0.5], [0, 0, 0], client=self.client, ctrl_type=UavCtrlType.VEL
        )

        plotter = UavLogger(ctrl_type=UavCtrlType.VEL)
        plotter.add_uav(self.uav.id)
        action = np.zeros(4)
        action[3]= self.uav.vel_lim
        # this determines how fast to complete a circle
        circ_freq = 1.0 / (240.0 * 2.0) * 2.0 * np.pi  # hz
        circ_rad = 0.9 * 100
        for i in range(10 * 240):
            action[0] = circ_rad * -np.sin(circ_freq * i) * circ_freq
            action[1] = circ_rad * np.cos(circ_freq * i) * circ_freq
            action[2] = 0.25

            self.uav.step(action)
            p.stepSimulation()
            plotter.log(
                uav_id=self.uav.id, state=self.uav.state.copy(), action=action.copy()
            )

        plotter.plot(title="Test Helix Using Velocity Tracking", plt_action=True)


if __name__ == "__main__":
    unittest.main()
