import logging

logger = logging.getLogger(__name__)

from rl_mus.envs.rl_mus import RlMus


class RlMusTtc(RlMus):
    def __init__(self, env_config=..., render_mode=None):
        super().__init__(env_config, render_mode)

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

        # # give penalty for reaching the time limit
        # if self.time_elapsed >= self.max_time:
        #     reward -= self._stp_penalty
        #     uav.done = True
        #     return reward

        uav.done_dt = t_remaining

        reward += max(0, 2 - uav.rel_target_dist**4)
        # reward += -0.3 * uav.rel_target_dist

        if uav.rel_target_dist <= 0.15:
            uav.target_reached = True

        # pos reward if uav reaches target
        if is_reached:
            uav.done = True
            uav.target_reached = True
            # reward += self._tgt_reward
            uav.terminated = True

            # # get reward for reaching destination in time
            # if abs(uav.done_dt) < self.t_go_max:
            #     reward += self._tgt_reward

            # else:
            #     reward += (
            #         -(1 - (self.time_elapsed / self.time_final)) * self._stp_penalty
            #     )

            # No need to check for other reward, UAV is done.
            return reward

        if (
            abs(uav.pos[0]) > self.env_max_l + 0.1
            or abs(uav.pos[1]) > self.env_max_w + 0.1
            or uav.pos[2] > self.env_max_h + 0.1
            or abs(uav.rpy[0]) > 0.4
            or abs(uav.rpy[1]) > 0.4
        ):
            uav.truncated = True
            uav.done = True
            uav.crashed = True
            reward += -self._crash_penalty
            return reward

        # if uav.pos[2] <= 0.02:
        #     # reward += -self._crash_penalty
        #     uav.truncated = True
        #     uav.crashed = True

        #     return reward

        # else:
        #     reward += -self._beta * (
        #         uav.rel_target_dist
        #         / np.linalg.norm(
        #             [2 * self.env_max_l, 2 * self.env_max_w, self.env_max_h]
        #         )
        #     )

        # if uav.rel_target_dist >= np.linalg.norm(
        #     [self.env_max_l, self.env_max_w, self.env_max_h]
        # ):
        #     reward += -10

        # reward += (
        #     -self._beta
        #     * uav.rel_target_dist
        #     / np.linalg.norm([self.env_max_l, self.env_max_w, self.env_max_h])
        # )

        # reward += -uav.rel_target_dist

        # reward += -3 * uav.los_angle(target) / np.pi

        # give small penalty for having large relative velocity
        # reward += -self._beta * uav.rel_target_vel

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
        if self.num_obstacles == 0:
            num_obstacle_shape = 6
        else:
            num_obstacle_shape = self.obstacles[0].state.shape[0]

        num_uav_state_shape = self.uavs[self.first_uav_id].state.shape
        action_buffer_size = self.uavs[self.first_uav_id].action_buffer_size

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
                        # "done_dt": spaces.Box(
                        #     low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                        # ),
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
                        # "los_angle": spaces.Box(
                        #     low=-np.pi,
                        #     high=np.pi,
                        #     shape=(1,),
                        #     dtype=np.float32
                        # )
                        # "action_buffer": spaces.Box(
                        #     low=self.action_low,
                        #     high=self.action_high,
                        #     shape=(action_buffer_size, self.num_actions),
                        #     dtype=np.float32,
                        # ),
                        # "constraint": spaces.Box(
                        #     low=-np.inf,
                        #     high=np.inf,
                        #     shape=(self.num_uavs - 1 + self.num_obstacles,),
                        #     dtype=np.float32,
                        # ),
                        # # TODO: need to fix for custom network
                        # "other_uav_obs": spaces.Box(
                        #     low=-np.inf,
                        #     high=np.inf,
                        #     shape=((self.num_uavs - 1) * num_uav_state_shape[0],),
                        #     dtype=np.float32,
                        # ),
                        # # "obstacles": spaces.Box(
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
    
    def _get_obs(self, uav):
        return super()._get_obs(uav)
