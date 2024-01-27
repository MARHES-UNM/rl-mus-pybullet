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
    def __init__(self, init_xyz, init_rpy, client, urdf="box.urdf", _type=AgentType.O):
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
        self.ctrl_type = ctrl_type
        super().__init__(init_xyz, init_rpy, client, urdf, _type)

    def _get_kinematic(self):
        self.pos, self.quat = p.getBasePositionAndOrientation(
            self.id, physicsClientId=self.client
        )
        self.rpy = p.getEulerFromQuaternion(self.quat)
        self.vel, self.ang_v = p.getBaseVelocity(self.id, physicsClientId=self.client)

    def get_rpm_from_action(action):
        pass

    def step(self, action=np.zeros(4)):

        kf = 3.16e-10
        km = 7.94e-12
        if self.ctrl_type == UavCtrlType.VEL:
            self.get_rpm_from_action(action)
        else:
            rpms = np.square(action)

        forces = rpms * kf
        torques = rpms * km
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
