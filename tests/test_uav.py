import unittest

import pybullet as p
import pybullet_data
from rl_mus.agents.uav import Entity, Uav, UavCtrlConf, UavCtrlType
from rl_mus.utils.plot_utils import Plotter, plot_traj
import time
import numpy as np
from rl_mus.utils.math_utils import wrap_angle


class TestUav(unittest.TestCase):
    def setUp(self) -> None:
        use_gui = True
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

            # axis_length = 2*self.l
            # self.x_axis = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
            #                                           lineToXYZ=[axis_length, 0, 0],
            #                                           lineColorRGB=[1, 0, 0],
            #                                           parentObjectUniqueId=self.uav.id,
            #                                           parentLinkIndex=-1,
            #                                           replaceItemUniqueId=int(self.X_AX[nth_drone]),
            #                                           physicsClientId=self.client
            #                                           )
            # self.y_axis = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
            #                                           lineToXYZ=[0, axis_length, 0],
            #                                           lineColorRGB=[0, 1, 0],
            #                                           parentObjectUniqueId=self.uav.id,
            #                                           parentLinkIndex=-1,
            #                                           replaceItemUniqueId=int(self.Y_AX[nth_drone]),
            #                                           physicsClientId=self.client
            #                                           )
            # self.z_axis = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
            #                                           lineToXYZ=[0, 0, axis_length],
            #                                           lineColorRGB=[0, 0, 1],
            #                                           parentObjectUniqueId=self.uav.id,
            #                                           parentLinkIndex=-1,
            #                                           replaceItemUniqueId=int(self.Z_AX[nth_drone]),
            #                                           physicsClientId=self.client
            #                                           )
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

    def tearDown(self) -> None:
        p.disconnect()

    @unittest.skip
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

    @unittest.skip
    def test_uav_hover_rpm(self):
        uav_des_traj = []
        uav_trajectory = []

        start_pos = [0, 0, 1]
        start_rpy = [0, 0, 0]
        self.uav = Uav(
            start_pos, start_rpy, client=self.client, ctrl_type=UavCtrlType.RPM
        )
        rpms_des = np.zeros(4)

        des_pos = self.uav.state.copy()

        for i in range(2 * 240):
            self.uav.step(rpms_des)
            p.stepSimulation()

            uav_des_traj.append(des_pos.copy())
            uav_trajectory.append(self.uav.state.copy())

        uav_des_traj = np.array(uav_des_traj)
        uav_trajectory = np.array(uav_trajectory)

        plot_traj(uav_des_traj, uav_trajectory, title="Test Desired Controller")

    @unittest.skip
    def test_uav_hover_vel(self):
        vel_des = np.zeros(3)
        self.uav = Uav(
            [1, 0, 1], [0, 0, 0], client=self.client, ctrl_type=UavCtrlType.VEL
        )

        plotter = Plotter()
        plotter.add_uav(self.uav.id)

        for i in range(10 * 240):
            self.uav.step(vel_des)
            p.stepSimulation()

            plotter.log(self.uav.id, self.uav.state, vel_des)

        plotter.plot("Test UAV Hover Velocity")

    @unittest.skip
    def test_uav_vel_tracking(self):
        self.uav = Uav(
            [3, 2, 1], [0, 0, 0], client=self.client, ctrl_type=UavCtrlType.VEL
        )

        plotter = Plotter()
        plotter.add_uav(self.uav.id)

        vel_des = np.zeros(3)
        for i in range(10 * 240):
            if i < 1 * 240:
                pass
            elif i >= 1 * 240 and i < 2 * 240:
                vel_des = np.array([0, 0, 1.0])

            elif i >= 2 * 240 and i < 3 * 240:
                vel_des = np.array([0, 1.0, 0])

            elif i >= 3 * 240 and i < 4 * 240:
                vel_des = np.array([1.0, 0, 0])

            elif i >= 4 * 240 and i < 5 * 240:
                vel_des = np.array([1.0, 1.0, 1.0])

            elif i >= 6 * 240 and i < 7 * 240:
                vel_des = np.array([-1.0, 0.0, 0.0])

            elif i >= 7 * 240 and i < 8 * 240:
                vel_des = np.array([0.0, -0.5, 0.0])

            elif i >= 8 * 240 and i < 9 * 240:
                vel_des = np.array([0.0, 0.0, -0.5])

            elif i >= 9 * 240:
                vel_des = np.array([0, 0, 0])

            self.uav.step(vel_des)
            p.stepSimulation()

            plotter.log(self.uav.id, state=self.uav.state, ref_ctrl=vel_des)

        plotter.plot(title="Test UAV velocity control", plt_ctrl=True)

    @unittest.skip
    def test_uav_rand_vel_tracking(self):
        self.uav = Uav(
            [0, 0, 0.5], [0, 0, 0], client=self.client, ctrl_type=UavCtrlType.VEL
        )

        plotter = Plotter()
        plotter.add_uav(self.uav.id)

        time_to_change_vel = 2 * 240  # every 2 secs
        vel_des = np.zeros(3)
        max_vel = self.uav.vel_lim * 0.5
        for i in range(10 * 240):
            if i % time_to_change_vel == 0:
                vel_des = np.random.uniform(low=-max_vel, high=max_vel, size=(3,))

            self.uav.step(vel_des)
            p.stepSimulation()
            plotter.log(uav_id=self.uav.id, state=self.uav.state, ref_ctrl=vel_des)

        plotter.plot(title="Test Random Velocity Tracking", plt_ctrl=True)

    @unittest.skip
    def test_uav_circular_traj(self):
        """
        https://pressbooks.online.ucf.edu/phy2048tjb/chapter/4-4-uniform-circular-motion/
        """
        self.uav = Uav(
            [0, 0, 2], [0, 0, 0], client=self.client, ctrl_type=UavCtrlType.POS
        )


        axis_length = 2*self.uav.arm
        self.x_axis = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                    lineToXYZ=[axis_length, 0, 0],
                                                    lineColorRGB=[1, 0, 0],
                                                    parentObjectUniqueId=self.uav.id,
                                                    parentLinkIndex=-1,
                                                    replaceItemUniqueId=-1,
                                                    physicsClientId=self.client
                                                    )
        self.y_axis = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                    lineToXYZ=[0, axis_length, 0],
                                                    lineColorRGB=[0, 1, 0],
                                                    parentObjectUniqueId=self.uav.id,
                                                    parentLinkIndex=-1,
                                                    replaceItemUniqueId=-1,
                                                    physicsClientId=self.client
                                                    )
        self.z_axis = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                    lineToXYZ=[0, 0, axis_length],
                                                    lineColorRGB=[0, 0, 1],
                                                    parentObjectUniqueId=self.uav.id,
                                                    parentLinkIndex=-1,
                                                    replaceItemUniqueId=-1,
                                                    physicsClientId=self.client
                                                    )
        
        plotter = Plotter(ctrl_type=UavCtrlType.POS)
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
                uav_id=self.uav.id, state=self.uav.state.copy(), ref_ctrl=action.copy()
            )

        plotter.plot(title="Test Circular Velocity Tracking", plt_ctrl=True)

    def test_uav_helix_vel_tracking(self):
        self.uav = Uav(
            [0, 0, 0.5], [0, 0, 0], client=self.client, ctrl_type=UavCtrlType.VEL
        )

        plotter = Plotter(ctrl_type=UavCtrlType.VEL)
        plotter.add_uav(self.uav.id)
        action = np.zeros(3)
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
                uav_id=self.uav.id, state=self.uav.state.copy(), ref_ctrl=action.copy()
            )

        plotter.plot(title="Test Helix Using Velocity Tracking", plt_ctrl=True)


if __name__ == "__main__":
    unittest.main()
