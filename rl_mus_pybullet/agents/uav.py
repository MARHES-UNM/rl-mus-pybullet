import numpy as np
import os

# from sim.utils.utils import cir_traj, distance, angle, lqr
from enum import IntEnum
from pathlib import Path
import pybullet as p
import pybullet_data
import numpy as np
from rl_mus_pybullet.assets import ASSET_PATH

script_path = Path(__file__).parent.absolute().resolve()


class AgentType(IntEnum):
    U = 0  # uav
    O = 1  # obstacle
    T = 2  # target


class ObsType(IntEnum):
    S = 0  # Static
    M = 1  # Moving


class UavCtrlType(IntEnum):
    RPM = 0
    ACC = 1
    VEL = 2


class Entity:
    def __init__(
        self, init_xyz, init_rpy, client, urdf="box.urdf", g=9.81, _type=AgentType.O
    ):
        self.client = client
        self.type = _type
        self.urdf = urdf

        self.id = p.loadURDF(
            os.path.join(ASSET_PATH, self.urdf),
            init_xyz,
            p.getQuaternionFromEuler(init_rpy),
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=client,
        )

        self._get_kinematic()

    def _get_kinematic(self):
        self.pos, self.quat = p.getBasePositionAndOrientation(
            self.id, physicsClientId=self.client
        )
        self.rpy = p.getEulerFromQuaternion(self.quat)
        self.vel, self.ang_v = p.getBaseVelocity(self.id, physicsClientId=self.client)

    def step(self):
        raise NotImplemented()

    @property
    def state(self):
        self._state = np.hstack(
            [self.pos, self.quat, self.rpy, self.vel, self.ang_v]
        ).reshape(
            -1,
        )
        return self._state


class Uav(Entity):
    def __init__(
        self,
        init_xyz,
        init_rpy,
        client,
        urdf="cf2p.urdf",
        g=9.81,
        _type=AgentType.U,
        ctrl_type=UavCtrlType.RPM,
    ):
        super().__init__(init_xyz, init_rpy, client, urdf, g, _type)
        self.ctrl_type = ctrl_type

        self.m = 0.027
        self.arm = 0.0397
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

        self.hover_rpm = np.array([np.sqrt(self.g * self.m / (4 * self.kf))] * 4)

    def _get_kinematic(self):
        self.pos, self.quat = p.getBasePositionAndOrientation(
            self.id, physicsClientId=self.client
        )
        self.rpy = p.getEulerFromQuaternion(self.quat)
        self.vel, self.ang_v = p.getBaseVelocity(self.id, physicsClientId=self.client)

    def get_rpm_from_action(self, action):
        # self.P_COEFF_FOR = np.array([.4, .4, 1.25])
        # self.I_COEFF_FOR = np.array([.05, .05, .05])
        # self.D_COEFF_FOR = np.array([.2, .2, .5])
        # self.P_COEFF_TOR = np.array([70000., 70000., 60000.])
        # self.I_COEFF_TOR = np.array([.0, .0, 500.])
        # self.D_COEFF_TOR = np.array([20000., 20000., 12000.])
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

    def step(self, action=np.zeros(4)):
        if self.ctrl_type == UavCtrlType.VEL:
            rpms = self.get_rpm_from_action(action)
        else:
            rpms = action

        rpms_sq = np.square(rpms)
        forces = rpms_sq * self.kf
        torques = rpms_sq * self.km
        z_torque = -torques[0] + torques[1] - torques[2] + torques[3]
        for i in range(4):
            p.applyExternalForce(
                self.id,
                i,
                forceObj=[0, 0, forces[i]],
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=self.client,
            )
        p.applyExternalTorque(
            self.id,
            4,
            torqueObj=[0, 0, z_torque],
            flags=p.LINK_FRAME,
            physicsClientId=self.client,
        )

        self._get_kinematic()
