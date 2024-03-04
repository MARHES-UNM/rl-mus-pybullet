from logging import Logger
import time
from matplotlib import pyplot as plt
import numpy as np
import unittest
import context
from rl_mus.agents.agents import UavCtrlType

from rl_mus.envs.rl_mus_ttc import RlMusTtc
from rl_mus.utils.logger import UavLogger
from ray.rllib.utils import check_env
from rl_mus.utils.logger import EnvLogger


class TestRlMus(unittest.TestCase):
    def setUp(self):
        self.num_short_time = 2
        self.num_med_time = 4
        self.num_long_time = 10

    def test_check_env_single(self):
        check_env(RlMusTtc({"num_uavs": 1}))

    def test_check_env(self):
        check_env(RlMusTtc({}))

    # def test_observation_space(self):
    #     env = RlMus({"num_uavs": 1, "num_obstacles": 0})
    #     obs_space = env.observation_space
    #     self.assertEqual(len(obs_space.spaces), 1)
    #     self.assertEqual(obs_space[0]["obstacles"].shape[0], 0)
    #     self.assertEqual(obs_space[0]["other_uav_obs"].shape[0], 0)

    #     env = RlMus({"num_uavs": 1, "num_obstacles": 1})
    #     obs_space = env.observation_space
    #     self.assertEqual(len(obs_space.spaces), 1)
    #     self.assertEqual(obs_space[0]["obstacles"].shape[0], 1)
    #     self.assertEqual(obs_space[0]["other_uav_obs"].shape[0], 0)

    # env = RlMus({"num_uavs": 2, "num_obstacles": 0})
    # obs_space = env.observation_space
    # self.assertEqual(len(obs_space.spaces), 2)
    # self.assertEqual(obs_space[0]["obstacles"].shape[0], 0)
    # self.assertEqual(obs_space[0]["other_uav_obs"].shape[0], 1)

    #     env = RlMus({"num_uavs": 2, "num_obstacles": 1})
    #     obs_space = env.observation_space
    #     self.assertEqual(len(obs_space.spaces), 2)
    #     self.assertEqual(obs_space[0]["obstacles"].shape[0], 1)
    #     self.assertEqual(obs_space[0]["other_uav_obs"].shape[0], 1)

    #TODO: fix terminal time controller
    def test_time_coordinated_control(self):
        env = RlMusTtc(
            env_config={
                "num_uavs": 2,
                "renders": True,
                "env_freq": 30,
                "time_final": self.num_med_time,
                "t_go_max": 1.0,
                "d_thresh": 0.15
            }
        )
        log_config = {
            "obs_items": ["state", "target"],
            "info_items": [
                "obstacle_collision",
                "uav_rel_dist",
                "uav_rel_vel",
                "uav_collision",
                "uav_target_reached",
                "uav_done_dt",
            ],
            "log_reward": True,
            "log_freq": env.env_freq,
            "env_freq": env.env_freq,
            "logger_name": "Test TTC Controller",
        }

        env_logger = EnvLogger(num_uavs=env.num_uavs, log_config=log_config)

        obs, info = env.reset()
        num_timesteps = int(env.max_time * env.env_freq)

        eps_num = 0
        dt = env.sim_dt
        for time_step in range(num_timesteps):
            actions = {
                uav.id: np.zeros(
                    4,
                )
                for uav in env.uavs.values()
            }
            for uav in env.uavs.values():
                cur_vel = uav.vel
                acc = env.get_time_coord_action(uav)
                actions[uav.id][:3] = cur_vel + dt * acc

                actions[uav.id][3] = np.linalg.norm(actions[uav.id][:3])
            obs, reward, done, truncated, info = env.step(actions)

            if time_step % (env.env_freq / env_logger.log_freq) == 0:
                env_logger.log(
                    eps_num=eps_num, info=info, obs=obs, reward=reward, action=actions
                )

            env.render()

            if done["__all__"]:
                obs, info = env.reset()
                eps_num += 1

        self.assertEqual(
            env_logger.num_samples, num_timesteps / env.env_freq * env_logger.log_freq
        )

        env_logger.plot_env()
        env_logger.plot(plt_action=True, plt_target=True)

        env.close()

# Everything below is to make sure that the tests are run in a specific order.
def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestRlMus("test_check_env_single"))
    suite.addTest(TestRlMus("test_check_env"))
    suite.addTest(TestRlMus("test_time_coordinated_control"))

    return suite


if __name__ == "__main__":
    """
    verbosity determines the output from the test:
     0: quite, no output
     1: default
     2: verbose, get help string from the output
    """

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
