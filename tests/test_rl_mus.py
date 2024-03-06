from logging import Logger
import time
from matplotlib import pyplot as plt
import numpy as np
import unittest
import context
from rl_mus.agents.agents import UavCtrlType

from rl_mus.envs.rl_mus import RlMus
from rl_mus.utils.logger import UavLogger
from ray.rllib.utils import check_env
from rl_mus.utils.logger import EnvLogger


class TestRlMus(unittest.TestCase):
    def setUp(self):
        self.num_short_time = 2
        self.num_med_time = 4
        self.num_long_time = 10

    def test_check_env_single(self):
        check_env(RlMus({"num_uavs": 1}))

    def test_check_env(self):
        check_env(RlMus({}))

    def test_render(self):
        self.env = RlMus(env_config={"renders": True})
        obs, info = self.env.reset()
        for _ in range(100):
            self.env.render()

    def test_log_env(self):
        env = RlMus(env_config={"num_uavs": 1, "renders": False})
        log_config = {
            "obs_items": ["state", "target"],
            "info_items": ["uav_collision"],
            "log_freq": 10,
            "env_freq": env.env_freq,
        }

        env_logger = EnvLogger(num_uavs=env.num_uavs, log_config=log_config)

        obs, info = env.reset()
        num_seconds = self.num_med_time
        num_timesteps = num_seconds * env.env_freq

        eps_num = 0
        for time_step in range(num_timesteps):
            actions = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(actions)

            env_logger.log(
                eps_num=eps_num, info=info, obs=obs, reward=reward, action=actions
            )

            if done["__all__"]:
                obs, info = env.reset()



        env_logger.plot(plt_action=True)
        self.assertEqual(
            env_logger.num_samples, (num_timesteps) / env.env_freq * env_logger.log_freq
        )
    def test_random_action_sample(self):
        env = RlMus(env_config={"renders": True, "num_uavs": 4})
        obs, info = env.reset()

        num_timesteps = self.num_short_time * env.env_freq
        for i in range(num_timesteps):
            actions = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(actions)
            env.render()

    # TODO: troubleshoot position controller
    def test_uav_go_to_goal(self):
        env = RlMus(
            env_config={
                "renders": True,
                "num_uavs": 1,
                "uav_ctrl_type": UavCtrlType.POS,
                "target_pos_rand": False
            }
        )
        obs, info = env.reset()

        plotter = UavLogger(
            num_uavs=env.num_uavs, ctrl_type=env.uav_ctrl_type, log_freq=env.env_freq
        )

        num_timesteps = self.num_med_time * env.env_freq
        for i in range(num_timesteps):
            actions = {uav.id: np.zeros(4) for uav in env.uavs.values()}

            for uav in env.uavs.values():
                actions[uav.id][:3] = obs[uav.id]["target"]
                actions[uav.id][3] = obs[uav.id]["state"][9]

            obs, reward, done, truncated, info = env.step(actions)

            for uav in env.uavs.values():
                plotter.log(uav_id=uav.id, action=actions[uav.id], state=uav.state)

        plotter.plot(plt_action=True)
        env.close()

    def test_uav_hover(self):
        env = RlMus(
            env_config={
                "renders": True,
                "num_uavs": 4,
                "uav_ctrl_type": UavCtrlType.VEL,
            }
        )
        obs, info = env.reset()

        plotter = UavLogger(
            num_uavs=env.num_uavs, ctrl_type=env.uav_ctrl_type, log_freq=env.env_freq
        )

        actions = {uav.id: np.zeros(3) for uav in env.uavs.values()}
        num_timesteps = self.num_med_time * env.env_freq
        for i in range(num_timesteps):
            for uav in env.uavs.values():
                actions[uav.id] = np.zeros(3)

            obs, reward, done, truncated, info = env.step(actions)
            env.render()
            # time.sleep(1 / 240)

            for uav in env.uavs.values():
                plotter.log(uav_id=uav.id, action=actions[uav.id], state=uav.state)

        plotter.plot(plt_action=True)

        env.close()

    def test_uav_vel_control(self):
        env = RlMus(
            env_config={
                "renders": True,
                "num_uavs": 1,
                "uav_ctrl_type": UavCtrlType.VEL,
            }
        )
        obs, info = env.reset()

        plotter = UavLogger(
            num_uavs=env.num_uavs, ctrl_type=env.uav_ctrl_type, log_freq=env.env_freq
        )

        actions = {uav.id: np.zeros(3) for uav in env.uavs.values()}
        num_timesteps = self.num_med_time * env.env_freq
        for i in range(num_timesteps):
            for uav in env.uavs.values():
                if i % (env.env_freq) == 0:
                    actions = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(actions)
            env.render()
            time.sleep(1 / 240)

            for uav in env.uavs.values():
                plotter.log(uav_id=uav.id, action=actions[uav.id], state=uav.state)

        plotter.plot(plt_action=True)

        env.close()

    def apf_uav_controller(self, agent, target, ka=1):
        agent_pos = agent.pos
        target_pos = target.pos
        alpha = 0

        dist_to_target = agent.rel_dist(target) + 0.001

        target_star = 1 * (target.rad + agent.rad)
        if dist_to_target <= target_star:
            #     des_v = -ka * (agent_pos - target_pos)
            des_v = np.zeros(3)

        else:
            des_v = (
                -ka
                * (1 / dist_to_target**alpha)
                * ((agent_pos - target_pos) / dist_to_target)
            )

        return des_v

    def test_uav_apf_vel_control_single(self):

        env = RlMus(
            env_config={
                "num_uavs": 1,
                "renders": True,
                # "pybullet_freq": 240,
                # "env_freq": 48,
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
        }

        env_logger = EnvLogger(num_uavs=env.num_uavs, log_config=log_config)

        obs, info = env.reset()
        num_seconds = self.num_med_time
        num_timesteps = num_seconds * env.env_freq

        eps_num = 0
        for time_step in range(num_timesteps):
            actions = {
                uav.id: np.zeros(
                    4,
                )
                for uav in env.uavs.values()
            }

            for uav in env.uavs.values():
                actions[uav.id][3] = env.action_high
                des_v = self.apf_uav_controller(uav, env.targets[uav.target_id])
                actions[uav.id][:3] = des_v

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

    def test_uav_apf_vel_control(self):

        env = RlMus(
            env_config={
                "num_uavs": 4,
                "renders": True,
                "env_freq": 30,
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
        }

        env_logger = EnvLogger(num_uavs=env.num_uavs, log_config=log_config)

        obs, info = env.reset()
        num_seconds = self.num_med_time
        num_timesteps = num_seconds * env.env_freq

        eps_num = 0
        for time_step in range(num_timesteps):
            actions = {
                uav.id: np.zeros(
                    4,
                )
                for uav in env.uavs.values()
            }
            for uav in env.uavs.values():
                actions[uav.id][3] = env.action_high
                des_v = self.apf_uav_controller(uav, env.targets[uav.target_id], ka=1)
                actions[uav.id][:3] = des_v

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
    # suite.addTest(TestRlMus("test_check_env_single"))
    # suite.addTest(TestRlMus("test_check_env"))
    # suite.addTest(TestRlMus("test_log_env"))
    # suite.addTest(TestRlMus("test_uav_go_to_goal"))
    # suite.addTest(TestRlMus("test_uav_vel_control"))
    suite.addTest(TestRlMus("test_uav_apf_vel_control"))
    suite.addTest(TestRlMus("test_uav_apf_vel_control_single"))

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
