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


class Uav:
    def __init__(
        self, init_xyz, init_rpy, client, urdf="cf2p.urdf", g=9.81, _type=AgentType.U
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

        # self._state = np.array([self.x, self.y, self.z, 0, 0, 0])
        self._get_kinematic()

    def _get_kinematic(self):
        self.pos, self.quat = p.getBasePositionAndOrientation(
            self.id, physicsClientId=self.client
        )
        self.rpy = p.getEulerFromQuaternion(self.quat)
        self.vel, self.ang_v = p.getBaseVelocity(self.id, physicsClientId=self.client)

    def step(self, action=None):
        kf = 3.16e-10
        km = 7.94e-12
        g = 9.81
        m = 0.027
        rpm = np.sqrt(g * m / (4 * kf))
        forces = np.array([rpm**2] * 4) * kf
        torques = np.array([rpm**2] * 4) * km

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

    @property
    def state(self):
        self._state = np.hstack(
            [self.pos, self.quat, self.rpy, self.vel, self.ang_v]
        ).reshape(
            -1,
        )
        return self._state

    # def in_collision(self, entity):
    #     dist = np.linalg.norm(self._state[0:3] - entity.state[0:3])
    #     return dist <= (self.r + entity.r)

    # def wrap_angle(self, val):
    #     return (val + np.pi) % (2 * np.pi) - np.pi

    # def rel_distance(self, entity):
    #     dist = np.linalg.norm(self._state[0:3] - entity.state[0:3])
    #     return dist

    # def rel_vel(self, entity):
    #     vel = np.linalg.norm(self._state[3:6] - entity.state[3:6])
    #     return vel


# class Obstacle(Entity):
#     def __init__(self, _id, x=0, y=0, z=0, r=0.1, dt=0.1, _type=ObsType.S):
#         super().__init__(_id, x, y, z, r, dt=dt, _type=_type)

#     def step(self, action=np.zeros(2)):
#         if self.type == ObsType.M:
#             vx = action[0] + np.random.random() * 0.2
#             vy = action[1] + np.random.random() * 0.2

#             self._state[3] = vx
#             self._state[4] = vy

#             self._state[0] += vx * self.dt
#             self._state[1] += vy * self.dt


# class Target(Entity):
#     def __init__(
#         self,
#     ):
#         super().__init__()

#     def step(self):
#         pass


# class UavBase(Entity):
#     def __init__(
#         self,
#         _id,
#         x=0,
#         y=0,
#         z=0,
#         r=0.1,
#         dt=1 / 10,
#         m=0.18,
#         l=0.086,
#         pad=None,
#         d_thresh=0.01,
#     ):
#         super().__init__(_id, x, y, z, r, _type=AgentType.U)

#         # timestep
#         self.dt = dt  # s

#         # gravity constant
#         self.g = 9.81  # m/s^2

#         self.m = m

#         # lenght of arms
#         self.l = l  # m

#         self._state = np.zeros(12)
#         self._state[0] = x
#         self._state[1] = y
#         self._state[2] = z
#         self.done = False
#         self.landed = False
#         self.pad = pad
#         self.done_time = None
#         self.d_thresh = d_thresh

#     def rk4(self, state, action):
#         """Based on: https://github.com/mahaitongdae/Safety_Index_Synthesis/blob/master/envs_and_models/collision_avoidance_env.py#L194
#         https://www.geeksforgeeks.org/runge-kutta-4th-order-method-solve-differential-equation/

#         Args:
#             state (_type_): _description_
#             action (_type_): _description_
#             dt (_type_): _description_

#         Returns:
#             _type_: _description_
#         """
#         dot_s1 = self.f_dot(state, action)
#         dot_s2 = self.f_dot(state + 0.5 * self.dt * dot_s1, action)
#         dot_s3 = self.f_dot(state + 0.5 * self.dt * dot_s2, action)
#         dot_s4 = self.f_dot(state + self.dt * dot_s3, action)
#         dot_s = (dot_s1 + 2 * dot_s2 + 2 * dot_s3 + dot_s4) / 6.0
#         return dot_s

#     def f_dot(self, state, action):
#         # copies action so we don't corrupt it.
#         temp_action = action.copy()
#         temp_action[2] = 1 / self.m * temp_action[2] - self.g

#         A = np.zeros((12, 12), dtype=np.float32)
#         A[0, 3] = 1.0
#         A[1, 4] = 1.0
#         A[2, 5] = 1.0

#         B = np.zeros((12, 3), dtype=np.float32)
#         B[3, 0] = 1.0
#         B[4, 1] = 1.0
#         B[5, 2] = 1.0

#         dxdt = A.dot(state) + B.dot(temp_action)

#         return dxdt

#     def rotation_matrix(self):
#         return np.eye(3)

#     def step(self, action=np.zeros(3)):
#         """Action is propeller forces in body frame

#         Args:
#             action (_type_, optional): _description_. Defaults to np.zeros(3).
#             state:
#             x, y, z, x_dot, y_dot, z_dot, phi, theta, psi, phi_dot, theta_dot, psi_dot
#         """
#         # keep uav hovering
#         action[2] = self.m * (self.g + action[2])

#         dot_state = self.rk4(self._state, action)
#         self._state = self._state + dot_state * self.dt

#         self._state[2] = max(0, self._state[2])

#     # # TODO: Combine the functions below into one
#     # def get_landed(self, pad):
#     #     dist = np.linalg.norm(self._state[0:3] - pad._state[0:3])
#     #     return dist <= 0.01

#     # TODO: combine into one.
#     def check_dest_reached(self, pad=None):
#         if pad is None:
#             pad = self.pad

#         rel_dist = np.linalg.norm(self._state[0:3] - pad.state[0:3])
#         rel_vel = np.linalg.norm(self._state[3:6] - pad.state[3:6])
#         # return rel_dist <= (self.r + pad.r), rel_dist, rel_vel
#         # TODO: set this to be a small number to make it more challenging
#         return rel_dist <= self.d_thresh, rel_dist, rel_vel

#     # TODO: combine with equations above
#     def get_rel_pad_dist(self):
#         return np.linalg.norm(self._state[:3] - self.pad._state[:3])

#     def get_rel_pad_vel(self):
#         return np.linalg.norm(self._state[3:6] - self.pad._state[3:6])

#     def get_t_go_est(self):
#         return self.get_rel_pad_dist() / (1e-6 + self.get_rel_pad_vel())


# class Uav(UavBase):
#     def __init__(
#         self,
#         _id,
#         x=0,
#         y=0,
#         z=0,
#         phi=0,
#         theta=0,
#         psi=0,
#         r=0.1,
#         dt=1 / 10,
#         m=0.18,
#         l=0.086,
#         k=None,
#         use_ode=False,
#         pad=None,
#         d_thresh=0.01,
#     ):
#         super().__init__(
#             _id=_id, x=x, y=y, z=z, r=r, dt=dt, m=m, l=l, pad=pad, d_thresh=d_thresh
#         )

#         self.use_ode = use_ode
#         if self.use_ode:
#             self.f_dot_ode = lambda time, state, action: self.f_dot(state, action)
#             self.ode = scipy.integrate.ode(self.f_dot_ode).set_integrator(
#                 "vode", nsteps=500, method="bdf"
#             )

#         self.inertia = np.array(
#             [[0.00025, 0, 2.55e-6], [0, 0.000232, 0], [2.55e-6, 0, 0.0003738]],
#             dtype=np.float64,
#         )

#         ## parameters from: https://upcommons.upc.edu/bitstream/handle/2117/187223/final-thesis.pdf?sequence=1&isAllowed=y
#         # self.inertia = np.eye(3)
#         # self.inertia[0, 0] = 0.0034  # kg*m^2
#         # self.inertia[1, 1] = 0.0034  # kg*m^2
#         # self.inertia[2, 2] = 0.006  # kg*m^2
#         # self.m = 0.698
#         self.ixx = self.inertia[0, 0]
#         self.iyy = self.inertia[1, 1]
#         self.izz = self.inertia[2, 2]

#         self.inv_inertia = np.linalg.pinv(self.inertia)

#         self.min_f = 0.0  # Neutons kg * m / s^2
#         self.max_f = 2.0 * self.m * self.g  # Neutons

#         # gamma = k_M / k_F
#         self.gamma = 1.5e-9 / 6.11e-8  # k_F = N / rpm^2, k_M = N*m / rpm^2

#         self._state = np.zeros(12)
#         self._state[0] = x
#         self._state[1] = y
#         self._state[2] = z
#         self._state[6] = phi
#         self._state[7] = theta
#         self._state[8] = psi
#         self.done = False
#         self.landed = False
#         self.pad = pad
#         self.done_time = None

#         # set up gain matrix
#         self.kx = self.ky = 1
#         self.kz = 1
#         self.k_x_dot = self.k_y_dot = 2
#         self.k_z_dot = 2
#         self.k_phi = 40
#         self.k_theta = 30
#         self.k_psi = 19
#         self.k_phi_dot = self.k_theta_dot = 5
#         self.k_psi_dot = 2

#         # set up gain matrix
#         self.kx = self.ky = 3.5
#         self.kz = 7
#         self.k_x_dot = self.k_y_dot = 3
#         self.k_z_dot = 4.5
#         self.k_phi = self.k_theta = 100
#         self.k_psi = 50
#         self.k_phi_dot = self.k_theta_dot = 15
#         self.k_psi_dot = 10

#     def get_r_matrix(self, phi, theta, psi):
#         """Calculates the Z-Y-X rotation matrix.
#            Based on Different Linearization Control Techniques for a Quadrotor System

#         Returns: R - 3 x 3 rotation matrix
#         """
#         cp = cos(phi)
#         sp = sin(phi)
#         ct = cos(theta)
#         st = sin(theta)
#         cg = cos(psi)
#         sg = sin(psi)
#         # R_x = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
#         # R_y = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
#         # R_z = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])

#         # R = np.dot(np.dot(R_x, R_y), R_z)
#         # ZXY matrix
#         R = np.array(
#             [
#                 [cg * ct - sp * sg * st, -cp * sg, cg * st + ct * sp * sg],
#                 [ct * sg + cg * sp * st, cp * cg, sg * st - cg * ct * sp],
#                 [-cp * st, sp, cp * ct],
#             ]
#         )
#         # R = np.array(
#         #     [
#         #         [cg * ct - sp * sg * st, ct * sg + cg * sp * st, -cp * st],
#         #         [-cp * sg, cp * cg, sp],
#         #         [cg * st + ct * sp * sg, sg * st - cg * ct * sp, cp * ct],
#         #     ]
#         # )
#         # R = R.transpose()
#         # R = np.dot(np.dot(R_z, R_x), R_y)
#         # R = np.dot(R_z, np.dot(R_y, R_x))
#         return R

#     def get_r_dot_matrix(self, phi, theta, psi):
#         """ """
#         cp = cos(phi)
#         sp = sin(phi)
#         ct = cos(theta)
#         st = sin(theta)
#         cg = cos(psi)
#         sg = sin(psi)

#         return np.array([[ct, 0, -cp * st], [0, 1, sp], [st, 0, cp * ct]])

#     def rotation_matrix(
#         self,
#     ):
#         """Calculates the Z-Y-X rotation matrix.
#            Based on Different Linearization Control Techniques for a Quadrotor System

#         Returns: R - 3 x 3 rotation matrix
#         """
#         cp = cos(self._state[6])
#         sp = sin(self._state[6])
#         ct = cos(self._state[7])
#         st = sin(self._state[7])
#         cg = cos(self._state[8])
#         sg = sin(self._state[8])
#         R_x = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
#         R_y = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
#         R_z = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])

#         # Z Y X
#         # R = np.dot(R_x, np.dot(R_y, R_z))
#         # # R = np.dot(np.dot(R_z, R_x), R_y)
#         # R = np.array(
#         #     [
#         #         [ct * cg, sp * st * cg - cp * sg, cp * st * cg + sp * sg],
#         #         [ct * sg, sp * st * sg + cp * cg, cp * st * sg - sp * cg],
#         #         [-st, sp * ct, cp * ct],
#         #     ]
#         # )
#         # R = np.array(
#         #     [
#         #         [cg * ct - sp * sg * st, -cp * sg, cg * st + ct * sp * sg],
#         #         [ct * sg + cg * sp * st, cp * cg, sg * st - cg * ct * sp],
#         #         [-cp * st, sp, cp * ct],
#         #     ]
#         # )
#         R = np.array(
#             [
#                 [cg * ct - sp * sg * st, ct * sg + cg * sp * st, -cp * st],
#                 [-cp * sg, cp * cg, sp],
#                 [cg * st + ct * sp * sg, sg * st - cg * ct * sp, cp * ct],
#             ]
#         )
#         R = np.dot(np.dot(R_z, R_x), R_y)
#         return R

#     def f_dot(self, state, action):
#         ft, tau_x, tau_y, tau_z = action.reshape(-1).tolist()

#         A = np.array(
#             [
#                 [0.25, 0, -0.5 / self.l],
#                 [0.25, 0.5 / self.l, 0],
#                 [0.25, 0, 0.5 / self.l],
#                 [0.25, -0.5 / self.l, 0],
#             ]
#         )
#         prop_thrusts = np.dot(A, np.array([ft, tau_x, tau_y]))
#         prop_thrusts_clamped = np.clip(prop_thrusts, self.min_f / 4.0, self.max_f / 4.0)

#         B = np.array(
#             [
#                 [1, 1, 1, 1],
#                 [0, self.l, 0, -self.l],
#                 [-self.l, 0, self.l, 0],
#             ]
#         )

#         ft = np.dot(B[0, :], prop_thrusts_clamped)
#         M = np.dot(B[1:, :], prop_thrusts_clamped)
#         tau = np.array([*M, tau_z])

#         # TODO: convert angular velocity to angle rates here:
#         # state[6:9] = self.wrap_angle(state[6:9])
#         phi = state[6]
#         theta = state[7]
#         psi = state[8]

#         omega = state[9:12].copy()
#         # tau = np.array([tau_x, tau_y, tau_z])

#         omega_dot = np.dot(
#             self.inv_inertia, (tau - np.cross(omega, np.dot(self.inertia, omega)))
#         )

#         # R = self.rotation_matrix()
#         # TODO: need to update the rotation matrix here using information from the ODE
#         R = self.get_r_matrix(phi, theta, psi)
#         acc = (
#             np.dot(R, np.array([0, 0, ft], dtype=np.float64).T)
#             - np.array([0, 0, self.m * self.g], dtype=np.float64).T
#         ) / self.m

#         # TODO: troubleshoot why we get small deviations in psi when doing this conversion
#         rot_dot = np.dot(np.linalg.inv(self.get_r_dot_matrix(phi, theta, psi)), omega)
#         # rot_dot = np.dot(self.get_r_dot_matrix(phi, theta, psi), omega)
#         # rot_dot = omega.copy()

#         # TODO: fix the x derivative matrix. This matrix doesn't provide angle rates
#         x_dot = np.array(
#             [
#                 state[3],
#                 state[4],
#                 state[5],
#                 acc[0],
#                 acc[1],
#                 acc[2],
#                 # TODO: use angle rates here instead
#                 rot_dot[0],
#                 rot_dot[1],
#                 rot_dot[2],
#                 omega_dot[0],
#                 omega_dot[1],
#                 omega_dot[2],
#             ]
#         )

#         return x_dot

#     def calc_des_action(self, des_pos):
#         pos_er = des_pos[0:12] - self._state
#         r_ddot_1 = des_pos[12]
#         r_ddot_2 = des_pos[13]
#         r_ddot_3 = des_pos[14]

#         action = np.zeros(3, dtype=np.float32)
#         # https://upcommons.upc.edu/bitstream/handle/2117/112404/Thesis-Jesus_Valle.pdf?sequence=1&isAllowed=y
#         action[0] = self.kx * pos_er[0] + self.k_x_dot * pos_er[3] + r_ddot_1
#         action[1] = self.ky * pos_er[1] + self.k_y_dot * pos_er[4] + r_ddot_2
#         action[2] = self.kz * pos_er[2] + self.k_z_dot * pos_er[5] + r_ddot_3

#         des_psi = des_pos[8]
#         des_psi_dot = des_pos[11]

#         des_action = self.get_torque_from_acc(action, des_psi, des_psi_dot)

#         return des_action

#     def get_torque_from_acc(self, action, des_psi=0, des_psi_dot=0):
#         u1 = self.m * (self.g + action[2])

#         # desired angles
#         phi_des = (action[0] * sin(des_psi) - action[1] * cos(des_psi)) / self.g
#         theta_des = (action[0] * cos(des_psi) + action[1] * sin(des_psi)) / self.g

#         # desired torques
#         u2_phi = self.k_phi * (phi_des - self._state[6]) + self.k_phi_dot * (
#             -self._state[9]
#         )
#         u2_theta = self.k_theta * (theta_des - self._state[7]) + self.k_theta_dot * (
#             -self._state[10]
#         )

#         # yaw
#         u2_psi = self.k_psi * (des_psi - self._state[8]) + self.k_psi_dot * (
#             des_psi_dot - self._state[11]
#         )

#         M = np.dot(self.inertia, np.array([u2_phi, u2_theta, u2_psi]))
#         action = np.array([u1, *M])
#         return action

#     def step(self, action=np.zeros(4)):
#         """Action is propeller forces in body frame

#         Args:
#             action (_type_, optional): _description_. Defaults to np.zeros(4).
#             state:
#             x, y, z, x_dot, y_dot, z_dot, phi, theta, psi, phi_dot, theta_dot, psi_dot
#         """

#         if len(action) == 3:
#             action = self.get_torque_from_acc(action)

#         if self.use_ode:
#             state = self._state.copy()
#             self.ode.set_initial_value(state, 0).set_f_params(action)
#             self._state = self.ode.integrate(self.ode.t + self.dt)
#         else:
#             dot_state = self.rk4(self._state, action)
#             self._state = self._state + dot_state * self.dt

#         self._state[9:12] = self.wrap_angle(self._state[9:12])

#         self._state[6:9] = self.wrap_angle(self._state[6:9])
#         self._state[2] = max(0, self._state[2])
