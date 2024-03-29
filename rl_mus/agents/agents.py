from collections import deque
import numpy as np
import os

from enum import IntEnum
from pathlib import Path
import pybullet as p
import pybullet_data
import numpy as np
from scipy.spatial.transform import Rotation
from rl_mus.assets import ASSET_PATH

script_path = Path(__file__).parent.absolute().resolve()


class AgentType(IntEnum):
    U = 0  # uav
    O = 1  # obstacle
    T = 2  # target


class ObsType(IntEnum):
    S = 0  # Static
    M = 1  # Moving
    T = 2  # Target


class UavCtrlType(IntEnum):
    RPM = 0
    ACC = 1
    VEL = 2
    POS = 3


class UavCtrlConf(IntEnum):
    X = 0
    P = 1


class Entity:
    def __init__(
        self,
        init_xyz,
        init_rpy,
        client,
        urdf="box.urdf",
        g=9.81,
        _type=AgentType.O,
        rad=0.1,
        show_local_axis=False,
        id=0,
    ):
        self.client = client
        self.type = _type
        self.urdf = urdf
        self.rad = rad
        self.show_local_axis = show_local_axis
        self.id = id

        self.p_id = p.loadURDF(
            self.urdf,
            init_xyz,
            p.getQuaternionFromEuler(init_rpy),
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=client,
        )

        if self.show_local_axis:
            self.draw_local_axis()

        self._get_kinematic()

    def _get_kinematic(self):
        self.pos, self.quat = p.getBasePositionAndOrientation(
            self.p_id, physicsClientId=self.client
        )
        self.rpy = np.array(p.getEulerFromQuaternion(self.quat))
        self.vel, self.ang_v = p.getBaseVelocity(self.p_id, physicsClientId=self.client)

        self.pos = np.array(self.pos)
        self.quat = np.array(self.quat)
        self.vel = np.array(self.vel)
        self.ang_v = np.array(self.ang_v)

        self._state = np.hstack(
            [self.pos, self.quat, self.rpy, self.vel, self.ang_v]
        ).reshape(
            -1,
        )

    def draw_local_axis(self):
        axis_length = 2 * self.rad
        self.x_axis = p.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=[axis_length, 0, 0],
            lineColorRGB=[1, 0, 0],
            parentObjectUniqueId=self.p_id,
            parentLinkIndex=-1,
            replaceItemUniqueId=-1,
            physicsClientId=self.client,
        )
        self.y_axis = p.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=[0, axis_length, 0],
            lineColorRGB=[0, 1, 0],
            parentObjectUniqueId=self.p_id,
            parentLinkIndex=-1,
            replaceItemUniqueId=-1,
            physicsClientId=self.client,
        )
        self.z_axis = p.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=[0, 0, axis_length],
            lineColorRGB=[0, 0, 1],
            parentObjectUniqueId=self.p_id,
            parentLinkIndex=-1,
            replaceItemUniqueId=-1,
            physicsClientId=self.client,
        )

    @property
    def state(self):
        return self._state

    def step(self):
        raise NotImplemented()

    def in_collision(
        self,
        entity,
        pos=None,
        rad=0.1,
    ):
        """
        #TODO: Find a better to represent collisions
        """
        if pos is None:
            dist = np.linalg.norm(self._state[0:3] - entity.state[0:3])
            return dist <= (self.rad + entity.rad)
        else:
            dist = np.linalg.norm(self.pos - pos)
            return dist <= (self.rad + rad)

    def rel_dist(self, entity):
        return np.linalg.norm(self.pos - entity.pos)

    def rel_vel(self, entity):
        return np.linalg.norm(self.vel - entity.vel)

    # TODO: validate this function
    def los_angle(self, entity):
        return np.arccos(
            np.dot(self.vel, (self.pos - entity.pos))
            / (np.linalg.norm(self.vel) * self.rel_dist(entity))
        )


class Target(Entity):
    def __init__(self, init_xyz, client, g=9.81, rad=0.1, show_local_axis=False, id=0):
        init_rpy = [0, 0, 0]
        _type = AgentType.T
        urdf = os.path.join(ASSET_PATH, "sphere.urdf")
        super().__init__(
            init_xyz,
            init_rpy,
            client=client,
            urdf=urdf,
            g=g,
            _type=_type,
            rad=rad,
            show_local_axis=show_local_axis,
            id=id,
        )

    def step(self, action=np.zeros(3)):
        pass


class Uav(Entity):
    def __init__(
        self,
        init_xyz,
        init_rpy,
        client,
        g=9.81,
        _type=AgentType.U,
        show_local_axis=False,
        ctrl_type=UavCtrlType.RPM,
        ctrl_conf=UavCtrlConf.X,
        pyb_freq=240.0,
        ctrl_freq=240.0,
        id=0,
    ):
        self.ctrl_conf = ctrl_conf
        self.ctrl_type = ctrl_type

        if self.ctrl_conf == UavCtrlConf.X:
            self.mixin_matrix = np.array(
                [[-0.5, -0.5, -1], [-0.5, 0.5, 1], [0.5, 0.5, -1], [0.5, -0.5, 1]]
            )
            urdf = "cf2x.urdf"
        elif self.ctrl_conf == UavCtrlConf.P:
            self.mixin_matrix = np.array(
                [[0, -1, -1], [+1, 0, 1], [0, 1, -1], [-1, 0, 1]]
            )
            urdf = "cf2p.urdf"
        else:
            raise TypeError("Unknow UAV configuration.")

        urdf = os.path.join(ASSET_PATH, urdf)
        self.arm = 0.0397

        super().__init__(
            init_xyz,
            init_rpy,
            client,
            urdf,
            g,
            _type,
            rad=self.arm,
            show_local_axis=show_local_axis,
            id=id,
        )

        self.m = 0.027
        self.kf = 3.16e-10
        self.km = 7.94e-12
        self.thrust2weight = 2.25
        self.max_speed_kmh = 30
        self.gnd_eff_coeff = 11.36859
        self.prop_radius = 2.31348e-2
        self.drag_coeff_xy = 9.1785e-7
        self.drag_coeff_z = 10.311e-7
        self.dw_coeff_1 = 2267.18
        self.dw_coeff_2 = 0.16
        self.dw_coeff_3 = -0.11
        self.g = g
        self.gravity = self.g * self.m
        # orig_scale = 0.12 makes 1 m/s, default 0.03
        self.vel_lim = 0.12 * self.max_speed_kmh * (1000 / 3600)  # 1 m/s
        self.pwm2rpm_scale = 0.2685
        self.pwm2rpm_const = 4070.3
        self.min_pwm = 20000
        self.max_pwm = 65535
        self.kp_for = np.array([0.4, 0.4, 1.25])
        self.ki_for = np.array([0.05, 0.05, 0.05])
        self.kd_for = np.array(
            [0.2, 0.2, 0.5]
        )  # gym-pybullet-drone uses .5 for z, 0.4 on board
        self.kp_tor = np.array([70000.0, 70000.0, 60000.0])
        self.ki_tor = np.array([0.0, 0.0, 500.0])
        self.kd_tor = np.array([20000.0, 20000.0, 12000.0])
        self.kd_omega_rp = np.array([200.0, 200.0, 0.0])

        # TODO: values below are from https://arxiv.org/pdf/2306.03951.pdf
        # self.kp_for = np.array([0.364, 0.364, 1.169])
        # self.ki_for = np.array([0.052, 0.052, 0.052])
        # self.kd_for = np.array(
        #     [0.234, 0.234, 0.586]
        # )  # gym-pybullet-drone uses .5 for z, 0.4 on board
        # self.kp_tor = np.array([64786.842, 64786.842, 55531.579])
        # self.ki_tor = np.array([0.0, 0.0, 599.666])
        # self.kd_tor = np.array([17217.406, 17217.406, 10330.44])
        # self.kd_omega_rp = np.array([200.0, 200.0, 0.0])

        self.last_ang_vel_des = np.zeros(3)
        self.last_ang_v = np.zeros(3)
        self.i_range_xy = 2.0
        self.i_range_z = 0.15  # on-board use .4
        self.i_range_m_xy = 1.0
        self.i_range_m_z = 1500.0
        self.pyb_freq = pyb_freq
        self.ctrl_freq = ctrl_freq
        self.ctrl_timestep = 1.0 / self.ctrl_freq
        self.rpms = np.zeros(4)
        self.last_rpy = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

        self.hover_rpm = np.array([np.sqrt((self.g * self.m) / (4 * self.kf))] * 4)

        self.target_id = None
        self.done = False
        self.target_reached = False
        self.crashed = False
        self.truncated = False
        self.terminated = False
        self.uav_collision = 0.0
        self.obs_collision = 0.0
        self.rel_target_dist = 0.0
        self.rel_target_vel = 0.0

        if self.ctrl_type == UavCtrlType.RPM:
            self.num_actions = 4
            self.action_low = np.ones(self.num_actions) * self.min_pwm
            self.action_high = np.ones(self.num_actions) * self.max_pwm
        elif self.ctrl_type == UavCtrlType.POS:
            self.num_actions = 4
            self.action_low = -np.ones(self.num_actions) * np.inf
            self.action_high = np.ones(self.num_actions) * np.inf
        elif self.ctrl_type == UavCtrlType.VEL:
            self.num_actions = 4
            self.action_low = -np.ones(self.num_actions) * self.vel_lim
            self.action_high = np.ones(self.num_actions) * self.vel_lim

        self.action_buffer_size = int(self.ctrl_freq // 2)
        self.action_buffer = deque(maxlen=self.action_buffer_size)
        for _ in range(self.action_buffer_size):
            self.action_buffer.append(np.zeros(self.num_actions))

    def compute_control(
        self, pos_des, rpy_des, vel_des=np.zeros(3), ang_vel_des=np.zeros(3)
    ):
        """Based on crazyflie mellinger controller:
        https://github.com/CrazyflieTHI/crazyflie-firmware/blob/master/src/modules/src/controller/controller_mellinger.c
        https://github.com/utiasDSL/gym-pybullet-drones/blob/main/gym_pybullet_drones/control/DSLPIDControl.py

        Args:
            pos_des (_type_): _description_
            rpy_des (_type_): _description_
            vel_des (_type_, optional): _description_. Defaults to np.zeros(3).
            ang_vel_des (_type_, optional): _description_. Defaults to np.zeros(3).

        Returns:
            _type_: _description_
        """
        thrust_des, comp_rpy_des = self.compute_position_control(
            pos_des, rpy_des, vel_des
        )

        rpm = self.compute_attitude_control(thrust_des, comp_rpy_des, ang_vel_des)

        return rpm

    def compute_position_control(self, pos_des, rpy_des, vel_des):
        rotation = np.array(p.getMatrixFromQuaternion(self.quat)).reshape(3, 3)
        pos_e = pos_des - self.pos
        vel_e = vel_des - self.vel

        self.integral_pos_e = self.integral_pos_e + pos_e * self.ctrl_timestep
        self.integral_pos_e = np.clip(
            self.integral_pos_e, -self.i_range_xy, self.i_range_xy
        )
        self.integral_pos_e[2] = np.clip(
            self.integral_pos_e[2], -self.i_range_z, self.i_range_z
        )
        thrust_des = (
            np.multiply(self.kp_for, pos_e)
            + np.multiply(self.ki_for, self.integral_pos_e)
            + np.multiply(self.kd_for, vel_e)
            + np.array([0, 0, self.gravity])
        )

        scalar_thrust = max(0.0, np.dot(thrust_des, rotation[:, 2]))
        scalar_thrust_des = (
            np.sqrt(scalar_thrust / (4.0 * self.kf)) - self.pwm2rpm_const
        ) / self.pwm2rpm_scale

        target_z_ax = thrust_des / np.linalg.norm(thrust_des)
        target_x_c = np.array([np.cos(rpy_des[2]), np.sin(rpy_des[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(
            np.cross(target_z_ax, target_x_c)
        )
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        rotation_des = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()

        comp_rpy_des = (Rotation.from_matrix(rotation_des)).as_euler(
            "XYZ", degrees=False
        )
        if np.any(np.abs(comp_rpy_des) > np.pi):
            print(
                "\n[ERROR] ctrl it",
                # self.control_counter,
                "in possition control, values outside range [-pi,pi]",
            )
        return scalar_thrust_des, comp_rpy_des

    def compute_attitude_control(self, thrust_des, rpy_des, ang_vel_des):
        rotation = np.array(p.getMatrixFromQuaternion(self.quat)).reshape(3, 3)

        quat_des = (Rotation.from_euler("XYZ", rpy_des, degrees=False)).as_quat()
        w, x, y, z = quat_des

        rotation_des = (Rotation.from_quat([w, x, y, z])).as_matrix()

        rot_matrix_e = np.dot((rotation_des.transpose()), rotation) - np.dot(
            rotation.transpose(), rotation_des
        )
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]])
        # ang_vel_e = ang_vel_des - self.ang_v
        rpy_rate = (self.rpy - self.last_rpy) / self.ctrl_timestep
        ang_vel_e = ang_vel_des - rpy_rate
        err_d_ang_vel = (
            (ang_vel_des - self.last_ang_vel_des) - (rpy_rate - self.last_ang_v)
        ) / self.ctrl_timestep

        self.last_ang_v = rpy_rate.copy()
        self.last_ang_vel_des = ang_vel_des.copy()
        self.last_rpy = self.rpy
        self.integral_rpy_e = self.integral_rpy_e - rot_e * self.ctrl_timestep
        self.integral_rpy_e = np.clip(
            self.integral_rpy_e, -self.i_range_m_z, self.i_range_m_z
        )
        self.integral_rpy_e[0:2] = np.clip(
            self.integral_rpy_e[0:2], -self.i_range_m_xy, self.i_range_m_xy
        )

        torques_des = (
            -np.multiply(self.kp_tor, rot_e)
            + np.multiply(self.ki_tor, self.integral_rpy_e)
            + np.multiply(self.kd_tor, ang_vel_e)
            # + np.multiply(self.kd_omega_rp, err_d_ang_vel)
        )

        torques_des = np.clip(torques_des, -3200, 3200)
        pwm = thrust_des + np.dot(self.mixin_matrix, torques_des)
        pwm = np.clip(pwm, self.min_pwm, self.max_pwm)

        return self.pwm2rpm_scale * pwm + self.pwm2rpm_const

    def get_rpm_from_action(self, action):
        """
        The GRASP Micro-UVA testbed
        https://ieeexplore.ieee.org/document/5569026
        """
        # TODO: Need to tune this controller
        kdx = 0.2
        kdy = 0.2
        kdz = 0.5
        kp_phi = kp_theta = 70000.0
        kp_psi = 60000.0
        kd_phi = kd_theta = 20000.0
        kd_psi = 12000.0

        vel_err = action - self.vel
        accx_des = kdx * vel_err[0]
        accy_des = kdy * vel_err[1]
        accz_des = kdz * vel_err[2]
        psi_des = self.rpy[2]
        single_hover_rpm = self.hover_rpm[0]

        mixin_matrix = np.array(
            [[1, 0, -1, 1], [1, 1, 0, -1], [1, 0, 1, 1], [1, -1, 0, -1]]
        )

        phi_des = 1 / self.g * (accx_des * np.sin(psi_des) - accy_des * np.cos(psi_des))
        theta_des = (
            1 / self.g * (accx_des * np.cos(psi_des) + accy_des * np.sin(psi_des))
        )

        d_omega_f = self.m / (8 * self.kf * single_hover_rpm) * accz_des

        d_omega_phi = kp_phi * (phi_des - self.rpy[0]) + kd_phi * (-self.ang_v[0])
        d_omega_theta = kp_theta * (theta_des - self.rpy[1]) + kd_theta * (
            -self.ang_v[1]
        )
        d_omega_psi = kp_psi * (psi_des - self.rpy[2]) + kd_psi * (-self.ang_v[2])

        rpms = np.dot(
            mixin_matrix,
            np.array(
                [d_omega_f + single_hover_rpm, d_omega_phi, d_omega_theta, d_omega_psi]
            ).T,
        )
        return rpms

    def preprocess_action(self, action):
        if self.ctrl_type == UavCtrlType.VEL:

            if np.linalg.norm(action[0:3]) != 0:
                vel_unit_vector = action[0:3] / np.linalg.norm(action[0:3])
            else:
                vel_unit_vector = np.zeros(3)

            rpms = self.compute_control(
                pos_des=self.pos,
                rpy_des=np.array([0, 0, self.rpy[2]]),
                vel_des=self.vel_lim
                * np.abs(action[3])
                * vel_unit_vector,  # target the desired velocity vector
            )

        elif self.ctrl_type == UavCtrlType.POS:
            rpms = self.compute_control(
                pos_des=action[0:3],
                rpy_des=np.array([0.0, 0.0, action[3]]),
            )
        # default is RPM control
        else:
            rpms = np.array(self.hover_rpm * (1 + 0.05 * action))

        return rpms

    def apf_control(self, target, ka=1):
        agent_pos = self.pos
        target_pos = target.pos
        alpha = 0

        dist_to_target = self.rel_dist(target) + 0.001

        target_star = 1 * (target.rad + self.rad)
        des_v = np.zeros(4)
        des_v[3] = self.vel_lim

        if dist_to_target <= target_star:
            des_v[:3] = np.zeros(3)

        else:
            des_v[:3] = (
                -ka
                * (1 / dist_to_target**alpha)
                * ((agent_pos - target_pos) / dist_to_target)
            )

        return des_v

    def step(self, action=np.zeros(4), preprocess_action=True):
        if preprocess_action:
            rpms = self.preprocess_action(action)
        else:
            rpms = action

        self.rpms = rpms

        rpms_sq = np.square(rpms)
        forces = rpms_sq * self.kf
        torques = rpms_sq * self.km
        z_torque = -torques[0] + torques[1] - torques[2] + torques[3]

        for i in range(4):
            p.applyExternalForce(
                self.p_id,
                i,
                forceObj=[0, 0, forces[i]],
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=self.client,
            )
        p.applyExternalTorque(
            self.p_id,
            4,
            torqueObj=[0, 0, z_torque],
            flags=p.LINK_FRAME,
            physicsClientId=self.client,
        )

        self._get_kinematic()

    @property
    def state(self):
        self._state = np.hstack(
            # [self.pos, self.quat, self.rpy, self.vel, self.ang_v, self.rpms]
            [self.pos, self.quat, self.rpy, self.vel, self.ang_v]
        ).reshape(
            -1,
        )
        return self._state
