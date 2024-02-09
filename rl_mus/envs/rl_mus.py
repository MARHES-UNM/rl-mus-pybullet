import sys
from gymnasium import spaces
import numpy as np
import gymnasium as gym
from gymnasium.utils import seeding
from rl_mus.agents.agents import Target, Uav, UavCtrlType
import logging
import random
from pybullet_utils import bullet_client as bc
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import io
import pybullet as p2
import pybullet_data


logger = logging.getLogger(__name__)


class RlMus(MultiAgentEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, env_config={}, render_mode=None):
        super().__init__()
        self.g = env_config.setdefault("g", 9.81)
        self._seed = env_config.setdefault("seed", None)
        self.render_mode = env_config.setdefault("render_mode", "human")
        self.num_uavs = env_config.setdefault("num_uavs", 4)
        self.gamma = env_config.setdefault("gamma", 1)
        self.num_obstacles = env_config.setdefault("num_obstacles", 0)
        self.obstacle_radius = env_config.setdefault("obstacle_radius", 1)
        self.max_num_obstacles = env_config.setdefault("max_num_obstacles", 0)
        assert self.max_num_obstacles >= self.num_obstacles, print(
            f"Max number of obstacles {self.max_num_obstacles} is less than number of obstacles {self.num_obstacles}"
        )
        self.obstacle_collision_weight = env_config.setdefault(
            "obstacle_collision_weight", 0.1
        )
        self.uav_collision_weight = env_config.setdefault("uav_collision_weight", 0.1)
        self._use_safe_action = env_config.setdefault("use_safe_action", False)
        self.time_final = env_config.setdefault("time_final", 20.0)
        self.t_go_max = env_config.setdefault("t_go_max", 2.0)
        self.t_go_n = env_config.setdefault("t_go_n", 1.0)
        self._beta = env_config.setdefault("beta", 0.01)
        self._d_thresh = env_config.setdefault("d_thresh", 0.15)  # uav.rad + target.rad
        self._tgt_reward = env_config.setdefault("tgt_reward", 100.0)
        self._stp_penalty = env_config.setdefault("stp_penalty", 100.0)
        self._dt_reward = env_config.setdefault("dt_reward", 0.0)
        self._dt_weight = env_config.setdefault("dt_weight", 0.0)

        self._uav_type = getattr(
            sys.modules[__name__], env_config.setdefault("uav_type", "Uav")
        )
        self.uav_ctrl_type = env_config.setdefault("uav_ctrl_type", UavCtrlType.VEL)

        self.env_max_w = env_config.setdefault("env_max_w", 4)
        self.env_max_l = env_config.setdefault("env_max_l", 4)
        self.env_max_h = env_config.setdefault("env_max_h", 4)
        self._z_high = env_config.setdefault("z_high", 4)
        self._z_low = env_config.setdefault("z_low", 0.1)
        self.max_time = self.time_final + (self.t_go_max * 2)

        self.env_config = env_config

        self._renders = env_config.setdefault("renders", False)
        self._render_height = env_config.setdefault("render_height", 200)
        self._render_width = env_config.setdefault("render_width", 320)
        self._pyb_freq = env_config.setdefault("pybullet_freq", 240)
        self._sim_dt = env_config.setdefault("sim_dt", 0.1)
        self._sim_steps = int(self._pyb_freq * self._sim_dt)
        # this is the timestep, default to 1 / 240
        self._pyb_dt = 1 / self._pyb_freq

        self._physics_client_id = None

        self._time_elapsed = 0.0
        self.seed(self._seed)
        self.reset()
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

    @property
    def time_elapsed(self):
        return self._time_elapsed

    @property
    def sim_dt(self):
        return self._sim_dt

    @property
    def env_freq(self):
        return int(1 / self._sim_dt)

    @property
    def pybullet_freq(self):
        return self._pyb_freq

    def _get_action_space(self):
        """The action of the UAV. We don't normalize the action space in this environment.
        It is recommended to normalize using a wrapper function.
        The uav action consist of acceleration in x, y, and z component."""
        return spaces.Dict(
            {
                uav.id: spaces.Box(
                    low=self.action_low,
                    high=self.action_high,
                    shape=(self.num_actions,),
                    dtype=np.float32,
                )
                for uav in self.uavs.values()
            }
        )

    def _get_observation_space(self):
        if self.num_obstacles == 0:
            num_obstacle_shape = 6
        else:
            num_obstacle_shape = self.obstacles[0].state.shape[0]

        num_uav_state_shape = self.uavs[self.first_uav_id].state.shape

        obs_space = spaces.Dict(
            {
                uav.id: spaces.Dict(
                    {
                        "state": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=num_uav_state_shape,
                            dtype=np.float32,
                        ),
                        "done_dt": spaces.Box(
                            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                        ),
                        "target": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(3,),
                            dtype=np.float32,
                        ),
                        "constraint": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(self.num_uavs - 1 + self.num_obstacles,),
                            dtype=np.float32,
                        ),
                        # TODO: need to fix for custom network
                        "other_uav_obs": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=((self.num_uavs - 1) * num_uav_state_shape[0],),
                            dtype=np.float32,
                        ),
                        # "obstacles": spaces.Box(
                        #     low=-np.inf,
                        #     high=np.inf,
                        #     shape=(
                        #         self.num_obstacles,
                        #         num_obstacle_shape,
                        #     ),
                        #     dtype=np.float32,
                        # ),
                    }
                )
                for uav in self.uavs.values()
            }
        )

        return obs_space

    def _get_uav_constraint(self, uav):
        """Return single uav constraint"""
        constraints = []

        for other_uav in self.uavs.values():
            if other_uav.id != uav.id:
                delta_p = uav.pos - other_uav.pos

                constraints.append(np.linalg.norm(delta_p) - (uav.rad + other_uav.rad))

        # TODO: handle obstacles
        # closest_obstacles = self._get_closest_obstacles(uav)

        # for obstacle in closest_obstacles:
        #     delta_p = uav.pos - obstacle.pos

        #     constraints.append(np.linalg.norm(delta_p) - (uav.rad + obstacle.rad))

        return np.array(constraints)

    def get_constraints(self):
        return {uav.id: self._get_uav_constraint(uav) for uav in self.uavs.values()}

    def get_h(self, uav, entity):
        del_p = uav.pos - entity.pos
        del_v = uav.vel - entity.vel

        h = np.linalg.norm(del_p) - (uav.rad + entity.rad)
        h = np.sqrt(h)
        h += (del_p.T @ del_v) / np.linalg.norm(del_p)

        return h

    def get_b(self, uav, entity):
        del_p = uav.pos - entity.pos
        del_v = uav.vel - entity.vel

        h = self.get_h(uav, entity)

        b = self.gamma * h**3 * np.linalg.norm(del_p)
        b -= ((del_v.T @ del_p) ** 2) / ((np.linalg.norm(del_p)) ** 2)
        b += (del_v.T @ del_p) / (
            np.sqrt(np.linalg.norm(del_p) - (uav.rad + entity.rad))
        )
        b += np.linalg.norm(del_v) ** 2
        return b

    def get_time_coord_action(self, uav):
        t = self.time_elapsed
        tf = self.time_final
        N = self.t_go_n

        des_pos = np.zeros(12)
        des_pos[0:6] = uav.pad.state[0:6]
        pos_er = des_pos - uav.state

        t0 = min(t, tf - 0.1)
        t_go = (tf - t0) ** N
        p = self.get_p_mat(tf, N, t0)
        B = np.zeros((2, 1))
        B[1, 0] = 1.0

        action = t_go * np.array(
            [
                B.T @ p[-1].reshape((2, 2)) @ pos_er[[0, 3]],
                B.T @ p[-1].reshape((2, 2)) @ pos_er[[1, 4]],
                B.T @ p[-1].reshape((2, 2)) @ pos_er[[2, 5]],
            ]
        )

        return action.squeeze()

    def get_tc_controller(self, uav):
        mean_tg_error = np.array(
            [
                x.get_t_go_est()  # - (self.time_final - self.time_elapsed)
                for x in self.uavs.values()
                if x.id != uav.id
            ]
        ).mean()
        cum_tg_error = (self.num_uavs / (self.num_uavs - 1)) * (
            # mean_tg_error - (uav.get_t_go_est() - (self.time_final - self.time_elapsed))
            mean_tg_error
            - (uav.get_t_go_est())  # - (self.time_final - self.time_elapsed)
        )
        cum_tg_error = 0

        for other_uav in self.uavs.values():
            if other_uav.id != uav.id:
                cum_tg_error += other_uav.get_t_go_est() - uav.get_t_go_est()

        des_pos = np.zeros(12)
        des_pos[0:6] = uav.pad.state[0:6]
        pos_er = des_pos - uav.state

        action = np.zeros(3)
        # if uav.id == 0:

        #     action = -1 * cum_tg_error * np.array([pos_er[0], pos_er[1], pos_er[2]])
        # else:
        #     action = (
        #         # -0.5 * cum_tg_error * np.array([pos_er[0], pos_er[1], pos_er[2]])
        #         -0.05
        #         * cum_tg_error
        #         * np.array([pos_er[0], pos_er[1], pos_er[2]])
        #     )
        # action = (
        #     # -0.5 * cum_tg_error * np.array([pos_er[0], pos_er[1], pos_er[2]])
        #     # -0.05
        #     # -5 / (uav.init_r * uav.init_tg)
        #     -0.5
        #     * cum_tg_error
        #     * np.array([pos_er[0], pos_er[1], pos_er[2]])
        # )
        cum_tg_error = self.time_final - self.time_elapsed
        action += (
            3 * np.array([pos_er[0], pos_er[1], pos_er[2]]) * (-0.3 * cum_tg_error)
        )

        # action += 2 * cum_tg_error * np.array([pos_er[3], pos_er[4], pos_er[5]])
        action += 3 * np.array([pos_er[3], pos_er[4], pos_er[5]])

        return action

    def get_p_mat(self, tf, N=1, t0=0.0):
        A = np.zeros((2, 2))
        A[0, 1] = 1.0

        B = np.zeros((2, 1))
        B[1, 0] = 1.0

        t_go = tf**N

        f1 = 2.0
        f2 = 2.0
        Qf = np.eye(2)
        Qf[0, 0] = f1
        Qf[1, 1] = f2

        Q = np.eye(2) * 0.0

        t = np.arange(tf, t0, -0.1)
        params = (tf, N, A, B, Q)

        g0 = np.array([*Qf.reshape((4,))])

        def dp_dt(time, state, tf, N, A, B, Q):
            t_go = (tf - time) ** N
            P = state[0:4].reshape((2, 2))
            p_dot = -(Q + P @ A + A.T @ P - P @ B * (t_go) @ B.T @ P)
            output = np.array(
                [
                    *p_dot.reshape((4,)),
                ]
            )
            return output

        result = odeint(dp_dt, g0, t, args=params, tfirst=True)
        return result

    def get_col_avoidance(self, uav, des_action):
        min_col_distance = uav.r * 2
        sum_distance = np.zeros(3)

        attractive_f = uav.pad.pos - uav.pos
        attractive_f = (
            self.action_high * attractive_f / (1e-3 + np.linalg.norm(attractive_f) ** 2)
        )

        # other agents
        for other_uav in self.uavs.values():
            if other_uav.id != uav.id:
                if uav.rel_distance(other_uav) <= (min_col_distance + other_uav.r):
                    dist = other_uav.pos - uav.pos
                    sum_distance += dist

        closest_obstacles = self._get_closest_obstacles(uav)

        for obstacle in closest_obstacles:
            if uav.rel_distance(obstacle) <= (min_col_distance + obstacle.r):
                dist = obstacle.pos - uav.pos
                sum_distance += dist

        dist_vect = np.linalg.norm(sum_distance)
        if dist_vect <= 1e-9:
            u_out = des_action.copy()
        else:
            u_out = -self.action_high * sum_distance / dist_vect
        return u_out

    def get_safe_action(self, uav, des_action):
        G = []
        h = []
        P = np.eye(3)
        u_in = des_action.copy()
        # u_in[2] = 1 / uav.m * u_in[2] - uav.g
        q = -np.dot(P.T, u_in)

        # other agents
        for other_uav in self.uavs.values():
            if other_uav.id != uav.id:
                G.append(-(uav.pos - other_uav.pos).T)
                b = self.get_b(uav, other_uav)
                h.append(b)

        for obstacle in self.obstacles:
            G.append(-(uav.pos - obstacle.pos).T)
            b = self.get_b(uav, obstacle)
            h.append(b)

        G = np.array(G)
        h = np.array(h)

        if G.any() and h.any():
            try:
                u_out = solve_qp(
                    P.astype(np.float64),
                    q.astype(np.float64),
                    G.astype(np.float64),
                    h.astype(np.float64),
                    None,
                    None,
                    None,
                    None,
                    solver="quadprog",
                )
            except Exception as e:
                print(f"error running solver: {e}")
                u_out = des_action
        else:
            print("not running qpsolver")
            return des_action

        if u_out is None:
            print("infeasible sovler")
            return des_action

        # if np.isclose(np.linalg.norm(u_out), 0.0) and not np.isclose(np.linalg.norm(des_action), 0):
        # print(f"uav_id: {uav.id} in deadlock")
        # if np.linalg.norm(des_action - u_out) > 1e-3:
        if np.linalg.norm(des_action - u_out) > 0.0001:
            # u_out += np.random.random(3)*.00001
            pass
            # print("safety layer in effect")

        # u_out[2] = uav.m * (uav.g + u_out[2])
        return u_out

    def step(self, actions):

        # step uavs
        self.alive_agents = set()

        # artificial step
        for num_step in range(self._sim_steps):
            for uav_id, action in actions.items():

                # # Done uavs don't move
                # if self.uavs[uav_id].done:
                #     continue

                self.alive_agents.add(uav_id)

                if self._use_safe_action:
                    action = self.get_safe_action(self.uavs[uav_id], action)

                # TODO: this may not be needed
                action = np.clip(action, self.action_low, self.action_high)
                self.uavs[uav_id].step(action)

            # step target
            for target in self.targets.values():
                target.step()

            # # step obstacles
            # for obstacle in self.obstacles:
            #     obstacle.step(np.array([self.target.vx, self.target.vy]))

            self._p.stepSimulation()
            self._time_elapsed += self._pyb_dt

        obs, reward, info = {}, {}, {}

        for uav_id in self.alive_agents:
            obs[uav_id] = self._get_obs(self.uavs[uav_id])
            reward[uav_id] = self._get_reward(self.uavs[uav_id])
            info[uav_id] = self._get_info(self.uavs[uav_id])

        # obs = {uav.id: self._get_obs(uav) for uav in self.uavs.values()}
        # reward = {uav.id: self._get_reward(uav) for uav in self.uavs.values()}
        # info = {uav.id: self._get_info(uav) for uav in self.uavs.values()}
        # obs = {id: self._get_obs(self.uavs[id]) for id in self.alive_agents}
        # reward = {id: self._get_reward(self.uavs[id]) for id in self.alive_agents}
        # info = {id: self._get_info(self.uavs[id]) for id in self.alive_agents}
        # calculate done for each agent
        # done = {self.uavs[id].id: self.uavs[id].done for id in self.alive_agents}
        # fake_done = {self.uavs[id].id: False for id in self.alive_agents}
        real_done = {self.uavs[id].id: self.uavs[id].done for id in self.alive_agents}
        real_done["__all__"] = (
            all(v for v in real_done.values()) or self._time_elapsed >= self.max_time
        )
        truncated = {
            self.uavs[id].id: self._time_elapsed >= self.max_time
            for id in self.alive_agents
        }
        truncated["__all__"] = all(v for v in truncated.values())

        # newwer api gymnasium > 0.28
        # return obs, reward, terminated, terminated, info

        # old api gym < 0.26.1
        # return obs, reward, done, info
        return obs, reward, real_done, real_done, info

    def _get_info(self, uav):
        """Must be called after _get_reward

        Returns:
            _type_: _description_
        """
        info = {
            "time_step": self.time_elapsed,
            "obstacle_collision": uav.obs_collision,
            "uav_rel_dist": uav.rel_target_dist,
            "uav_rel_vel": uav.rel_target_vel,
            "uav_collision": uav.uav_collision,
            "uav_target_reached": 1.0 if uav.target_reached else 0.0,
            "uav_done_dt": uav.done_dt,
        }

        return info

    def _get_closest_obstacles(self, uav):
        obstacle_states = np.array([obs.state for obs in self.obstacles])
        dist = np.linalg.norm(obstacle_states[:, :3] - uav.state[:3][None, :], axis=1)
        argsort = np.argsort(dist)[: self.num_obstacles]
        closest_obstacles = [self.obstacles[idx] for idx in argsort]
        return closest_obstacles

    def _get_obs(self, uav):
        other_uav_states = np.array(
            [
                other_uav.state
                for other_uav in self.uavs.values()
                if uav.id != other_uav.id
            ]
        )

        # TODO: handle obstacles
        # closest_obstacles = self._get_closest_obstacles(uav)
        # obstacles_to_add = np.array([obs.state for obs in closest_obstacles])

        obs_dict = {
            "state": uav.state.astype(np.float32),
            "target": self.targets[uav.target_id].pos.astype(np.float32),
            "done_dt": np.array(
                [self.time_final - self._time_elapsed], dtype=np.float32
            ),
            "other_uav_obs": other_uav_states.reshape(-1).astype(np.float32),
            # TODO: handle obstacles
            # "obstacles": obstacles_to_add.astype(np.float32),
            "constraint": self._get_uav_constraint(uav).astype(np.float32),
        }

        return obs_dict

    def _get_reward(self, uav):
        reward = 0.0
        target = self.targets[uav.target_id]
        t_remaining = self.time_final - self.time_elapsed
        uav.uav_collision = 0.0
        uav.obs_collision = 0.0
        uav.rel_target_dist = uav.rel_dist(target)
        uav.rel_target_vel = uav.rel_vel(target)
        is_reached = uav.rel_target_dist <= self._d_thresh

        if uav.done:
            # UAV most have finished last time_step, report zero collisions
            return reward

        # give penalty for reaching the time limit
        if self.time_elapsed >= self.max_time:
            reward -= self._stp_penalty
            uav.done = True
            return reward

        uav.done_dt = t_remaining

        # pos reward if uav reaches target
        if is_reached:
            uav.done = True
            uav.target_reached = True
            reward += self._tgt_reward

            # # get reward for reaching destination in time
            # if abs(uav.done_dt) < self.t_go_max:
            #     reward += self._tgt_reward

            # else:
            #     reward += (
            #         -(1 - (self.time_elapsed / self.time_final)) * self._stp_penalty
            #     )

            # No need to check for other reward, UAV is done.
            return reward

        elif uav.rel_target_dist >= np.linalg.norm(
            [self.env_max_l, self.env_max_w, self.env_max_h]
        ):
            reward += -10
        else:
            reward += -self._beta * (
                uav.rel_target_dist
                / np.linalg.norm([self.env_max_l, self.env_max_w, self.env_max_h])
            )
            # reward += -self._beta * uav.rel_target_dist

        if uav.pos[2] <= 0.02:
            reward += -10
            uav.done = True
        # give small penalty for having large relative velocity
        reward += -self._beta * uav.rel_target_vel

        # neg reward if uav collides with other uavs
        for other_uav in self.uavs.values():
            if uav.id != other_uav.id:
                if uav.in_collision(other_uav):
                    reward -= self.uav_collision_weight
                    uav.uav_collision += 1

        # neg reward if uav collides with obstacles
        for obstacle in self.obstacles:
            if uav.in_collision(obstacle):
                reward -= self.obstacle_collision_weight
                uav.obs_collision += 1

        return reward

    def seed(self, seed=None):
        """Random value to seed"""
        random.seed(seed)
        np.random.seed(seed)

        seed = seeding.np_random(seed)
        return [seed]

    def get_random_pos(
        self,
        low_h=0.1,
        x_high=None,
        y_high=None,
        z_high=None,
    ):
        if x_high is None:
            x_high = self.env_max_w
        if y_high is None:
            y_high = self.env_max_l
        if z_high is None:
            z_high = self.env_max_h

        x = np.random.rand() * x_high
        y = np.random.rand() * y_high
        z = np.random.uniform(low=low_h, high=z_high)
        return np.array([x, y, z])

    def is_in_collision(self, entity, pos, rad):
        for target in self.targets.values():
            if target.in_collision(entity, pos, rad):
                return True

        for obstacle in self.obstacles:
            if obstacle.in_collision(entity, pos, rad):
                return True

        for other_uav in self.uavs.values():
            if other_uav.in_collision(entity, pos, rad):
                return True

        return False

    def reset(self, *, seed=None, options=None):
        """_summary_

        Args:
            seed (_type_, optional): _description_. Defaults to None.
            options (_type_, optional): _description_. Defaults to None.
        """
        # not currently compatible with new gym api to pass in seed
        # if seed is None:
        #     seed = self._seed
        # super().reset(seed=seed)

        if self._physics_client_id is None:
            if self._renders:
                self._p = bc.BulletClient(connection_mode=p2.GUI)
            else:
                self._p = bc.BulletClient()

            self._physics_client_id = self._p._client

        p = self._p

        p.resetSimulation()

        if self._renders:
            for i in [
                p.COV_ENABLE_RGB_BUFFER_PREVIEW,
                p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
            ]:
                p.configureDebugVisualizer(
                    i, 0, physicsClientId=self._physics_client_id
                )
            p.resetDebugVisualizerCamera(
                cameraDistance=5,
                cameraYaw=-15,
                cameraPitch=-45,
                cameraTargetPosition=[0, 0, 0],
                physicsClientId=self._physics_client_id,
            )
            ret = p.getDebugVisualizerCamera(physicsClientId=self._physics_client_id)
            logger.debug("viewMatrix", ret[2])
            logger.debug("projectionMatrix", ret[3])

            # # ### Add input sliders to the GUI ##########################
            # # self.SLIDERS = -1*np.ones(4)
            # # for i in range(4):
            # # self.SLIDERS[i] = p.addUserDebugParameter("Propeller "+str(i)+" RPM", 0, self.MAX_RPM, self.HOVER_RPM, physicsClientId=self._physics_client_id)
            # # self.INPUT_SWITCH = p.addUserDebugParameter("Use GUI RPM", 9999, -1, 0, physicsClientId=self._physics_client_id)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        self.plane_id = p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -self.g)
        p.setTimeStep(self._pyb_dt)
        p.setRealTimeSimulation(0)

        p = self._p

        self._time_elapsed = 0.0
        self._agent_ids = set()

        # we need the first uav id for some housekeeping
        self.first_uav_id = None
        self.uavs = {}
        self.targets = {}
        self.obstacles = []

        # Reset the UAVs
        pos = [0.0, 0.0, 0.0]
        for idx in range(self.num_uavs):
            in_collision = True

            # make sure the agent is not in collision with other agents or obstacles
            # the lowest height needs to be the uav radius x2
            while in_collision:
                pos = self.get_random_pos(low_h=self._z_low, z_high=self._z_high)

                in_collision = self.is_in_collision(entity=None, pos=pos, rad=0.1)

            # position must be good if here
            uav = Uav(
                pos,
                [0, 0, 0],
                self._physics_client_id,
                g=self.g,
                ctrl_type=self.uav_ctrl_type,
                pyb_freq=self._pyb_freq,
            )
            if self.first_uav_id is None:
                self.first_uav_id = uav.id

            self._agent_ids.add(uav.id)

            self.uavs[uav.id] = uav

        # Reset Target
        for uav in self.uavs.values():
            # for _ in range(self.num_uavs):
            in_collision = True

            while in_collision:
                pos = self.get_random_pos(low_h=self._z_low, z_high=self._z_high)

                in_collision = self.is_in_collision(entity=None, pos=pos, rad=0.1)

            # position must be good if here
            target = Target(pos, self._physics_client_id, g=self.g)

            self.targets[target.id] = target
            uav.target_id = target.id

        # # Reset obstacles, obstacles should not be in collision with target. Obstacles can be in collision with each other.
        # self.obstacles = []
        # for idx in range(self.max_num_obstacles):
        #     in_collision = True

        #     while in_collision:
        #         x, y, z = get_random_pos(low_h=self.obstacle_radius * 1.50, z_high=3.5)
        #         _type = ObsType.S
        #         obstacle = Obstacle(
        #             _id=idx,
        #             x=x,
        #             y=y,
        #             z=z,
        #             r=self.obstacle_radius,
        #             dt=self.dt,
        #             _type=_type,
        #         )

        #         in_collision = any(
        #             [
        #                 obstacle.in_collision(other_obstacle)
        #                 for other_obstacle in self.obstacles
        #                 if obstacle.id != other_obstacle.id
        #             ]
        #         )

        #     self.obstacles.append(obstacle)

        self.norm_action_high = np.ones(3)
        self.norm_action_low = np.ones(3) * -1

        self.action_high = self.uavs[self.first_uav_id].action_high
        self.action_low = self.uavs[self.first_uav_id].action_low
        self.num_actions = self.uavs[self.first_uav_id].num_actions

        obs = {uav.id: self._get_obs(uav) for uav in self.uavs.values()}
        # we need to call reward here so that info items will get populated
        # reward = {uav.id: self._get_reward(uav) for uav in self.uavs.values()}
        # info = {uav.id: self._get_info(uav) for uav in self.uavs.values()}
        info = {uav.id: {} for uav in self.uavs.values()}

        return obs, info

    def unscale_action(self, action):
        """[summary]

        Args:
            action ([type]): [description]

        Returns:
            [type]: [description]
        """
        assert np.all(np.greater_equal(action, self.norm_action_low)), (
            action,
            self.norm_action_low,
        )
        assert np.all(np.less_equal(action, self.norm_action_high)), (
            action,
            self.norm_action_high,
        )
        action = self.action_low + (self.action_high - self.action_low) * (
            (action - self.norm_action_low)
            / (self.norm_action_high - self.norm_action_low)
        )
        # # TODO: this is not needed
        # action = np.clip(action, self.action_low, self.action_high)

        return action

    def scale_action(self, action):
        """Scale agent action between default norm action values"""
        # assert np.all(np.greater_equal(action, self.action_low)), (action, self.action_low)
        # assert np.all(np.less_equal(action, self.action_high)), (action, self.action_high)
        action = (self.norm_action_high - self.norm_action_low) * (
            (action - self.action_low) / (self.action_high - self.action_low)
        ) + self.norm_action_low

        return action

    def render(self, mode="human", close=False):
        """
        See this example for converting python figs to images:
        https://stackoverflow.com/questions/7821518/save-plot-to-numpy-array

        Args:
            mode (str, optional): _description_. Defaults to "human".
            done (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        if mode == "human":
            self._renders = True

        if mode != "rgb_array":
            return np.array([])

        base_pos = [0, 0, 0]
        self._cam_dist = 2
        self._cam_pitch = 0.3
        self._cam_yaw = 0
        if self._physics_client_id >= 0:
            view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=base_pos,
                distance=self._cam_dist,
                yaw=self._cam_yaw,
                pitch=self._cam_pitch,
                roll=0,
                upAxisIndex=2,
            )
            proj_matrix = self._p.computeProjectionMatrixFOV(
                fov=60,
                aspect=float(self._render_width) / self._render_height,
                nearVal=0.1,
                farVal=100.0,
            )
            (_, _, px, _, _) = self._p.getCameraImage(
                width=self._render_width,
                height=self._render_height,
                renderer=self._p.ER_BULLET_HARDWARE_OPENGL,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
            )
        else:
            px = np.array(
                [[[255, 255, 255, 255]] * self._render_width] * self._render_height,
                dtype=np.uint8,
            )
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(
            np.array(px), (self._render_height, self._render_width, -1)
        )
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        if self._physics_client_id is not None:
            self._p.disconnect()
        self._physics_client_id = None
        self.first_uav_id = None
