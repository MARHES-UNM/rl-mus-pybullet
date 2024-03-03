from gymnasium import spaces
import numpy as np
import logging

logger = logging.getLogger(__name__)

from rl_mus.envs.rl_mus import RlMus


class RlMusTtc(RlMus):
    def __init__(self, env_config=..., render_mode=None):
        super().__init__(env_config, render_mode)

    def _get_reward(self, uav):
        reward = 0.0
        t_remaining = self.time_final - self.time_elapsed
        uav.uav_collision = 0.0
        uav.obs_collision = 0.0

        if uav.done:
            # UAV most have finished last time_step, report zero collisions
            return reward

        # give penalty for reaching the time limit
        elif self.time_elapsed >= self.max_time:
            reward -= self._stp_penalty
            return reward

        uav.done_dt = t_remaining

        target = self.targets[uav.target_id]
        uav.rel_target_dist = uav.rel_dist(target)
        uav.rel_target_vel = uav.rel_vel(target)
        is_reached = uav.rel_target_dist <= self._d_thresh

        # pos reward if uav reaches target
        if is_reached:
            uav.done = True
            uav.target_reached = True
            uav.terminated = True

            # get reward for reaching destination in time
            if abs(uav.done_dt) < self.t_go_max:
                reward += self._tgt_reward

            else:
                reward += (
                    -(1 - (self.time_elapsed / self.time_final)) * self._stp_penalty
                )

            # No need to check for other reward, UAV is done.
            return reward

        elif uav.rel_target_dist >= np.linalg.norm(
            [2*self.env_max_w, 2*self.env_max_l, self.env_max_h]
        ):
            uav.crashed = True
            reward += -self._crash_penalty

        # elif (
        #     abs(uav.pos[0]) > self.env_max_l + 0.1
        #     or abs(uav.pos[1]) > self.env_max_w + 0.1
        #     or uav.pos[2] > self.env_max_h + 0.1
        #     or abs(uav.rpy[0]) > 0.4
        #     or abs(uav.rpy[1]) > 0.4
        # ):
        #     # uav.truncated = True
        #     # uav.done = True
        #     uav.crashed = True
        #     reward += -self._crash_penalty
        #     return reward
        else:
            reward -= self._beta * (
                uav.rel_target_dist
                / np.linalg.norm(
                    [2 * self.env_max_w, 2 * self.env_max_l, self.env_max_h]
                )
            )

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

    def _get_observation_space(self):
        num_obstacle_shape = 6
        num_uav_state_shape = 16

        obs_space = spaces.Dict(
            {
                uav_id: spaces.Dict(
                    {
                        "state": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(num_uav_state_shape,),
                            dtype=np.float32,
                        ),
                        "target": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(3,),
                            dtype=np.float32,
                        ),
                        "tgt_rel_dist": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(1,),
                            dtype=np.float32,
                        ),
                        "tgt_rel_vel": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(1,),
                            dtype=np.float32,
                        ),
                        # TODO: need to fix for custom network
                        # "other_uav_obs": spaces.Box(
                        #     low=-np.inf,
                        #     high=np.inf,
                        #     shape=((self.num_uavs - 1) * num_uav_state_shape,),
                        #     dtype=np.float32,
                        # ),
                        # "obstacles": spaces.Box(
                        #     low=-np.inf,
                        #     high=np.inf,
                        #     shape=(
                        #         self.num_obstacles,
                        #         num_obstacle_shape,
                        #     ),
                        #     dtype=np.float32,
                        # ),
                        "done_dt": spaces.Box(
                            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                        ),
                    }
                )
                for uav_id in range(self.num_uavs)
            }
        )

        return obs_space

    def _get_obs(self, uav):
        other_uav_states = np.array(
            [
                other_uav.state
                for other_uav in self.uavs.values()
                if uav.id != other_uav.id
            ]
        )
        target = self.targets[uav.target_id]
        uav.rel_target_dist = uav.rel_dist(target)
        uav.rel_target_vel = uav.rel_vel(target)

        # TODO: handle obstacles
        # closest_obstacles = self._get_closest_obstacles(uav)
        # obstacles_to_add = np.array([obs.state for obs in closest_obstacles])

        obs_dict = {
            "state": uav.state.astype(np.float32),
            "target": self.targets[uav.target_id].pos.astype(np.float32),
            "tgt_rel_dist": np.array([uav.rel_target_dist], dtype=np.float32),
            "tgt_rel_vel": np.array([uav.rel_target_vel], dtype=np.float32),
            "done_dt": np.array(
                [self.time_final - self._time_elapsed], dtype=np.float32
            ),
            # TODO: handle other uavs
            # "other_uav_obs": other_uav_states.reshape(-1).astype(np.float32),
            # TODO: handle obstacles
            # "obstacles": obstacles_to_add.astype(np.float32),

            # TODO: handle constraints
            # "constraint": self._get_uav_constraint(uav).astype(np.float32),
        }

        return obs_dict
