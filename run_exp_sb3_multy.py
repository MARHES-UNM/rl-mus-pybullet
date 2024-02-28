"""Script demonstrating the use of `gym_pybullet_drones`'s Gymnasium interface.

Classes HoverAviary and MultiHoverAviary are used as learning envs for the PPO algorithm.

Example
-------
In a terminal, run as:

    $ python learn.py --multiagent false
    $ python learn.py --multiagent true

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning library `stable-baselines3`.

"""

import json
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from stable_baselines3.common.evaluation import evaluate_policy

from rl_mus.envs.rl_mus import RlMus
from gym_pybullet_drones.utils.utils import sync, str2bool
from gymnasium.wrappers.flatten_observation import FlattenObservation
from rl_mus.utils.env_utils import get_git_hash
from rl_mus.utils.logger import EnvLogger


DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_COLAB = False

# DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
# DEFAULT_ACT = ActionType('vel') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 2
DEFAULT_MA = False

# env_cfg = {"num_uavs": 1, "seed": 123}
# env_cfg = {"num_uavs": 4, "renders": True}
env_cfg = {"num_uavs": 4, "renders": False, "seed": 123}
from gymnasium import spaces


# import gym
# from gym import spaces
# TODO: utilize this instead of gym.Wrapper
# https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/core.py
class RlMusFlattenObs(gym.ObservationWrapper):
    """Observation wrapper that flattens the observation.

    Example:
        >>> import gym
        >>> env = gym.make('CarRacing-v1')
        >>> env.observation_space.shape
        (96, 96, 3)
        >>> env = FlattenObservation(env)
        >>> env.observation_space.shape
        (27648,)
        >>> obs = env.reset()
        >>> obs.shape
        (27648,)
    """

    def __init__(self, env):
        """Flattens the observations of an environment.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.observation_space = spaces.flatten_space(env.observation_space)

    def observation(self, observation):
        """Flattens an observation.

        Args:
            observation: The observation to flatten

        Returns:
            The flattened observation
        """
        return spaces.flatten(self.env.observation_space, observation)


class RlMusFlatAct(gym.ActionWrapper):

    def __init__(self, env):
        self.uav_id = list(env.agent_ids)[0]
        super().__init__(env)

        self.action_space = spaces.flatten_space(env.action_space)

    def action(self, action):
        reshaped_action = action.copy().reshape(self.env.unwrapped.num_uavs, -1)
        actions = {}
        for idx, uav_id in enumerate(self.env.unwrapped.uavs.keys()):
            actions[uav_id] = reshaped_action[idx]

        return actions


class RlMusRewWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        return sum([rew for rew in reward.values()])


class RlMusTermWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        if "__all__" in term:
            term = term["__all__"]
        if "__all__" in trunc:
            trunc = trunc["__all__"]

        return obs, rew, term, trunc, info


env = RlMusTermWrapper(RlMusRewWrapper(RlMusFlatAct(RlMusFlattenObs(RlMus(env_cfg)))))

from stable_baselines3.common.env_checker import check_env

# It will check your custom environment and output additional warnings if needed
check_env(env)


obs, info = env.reset()

for i in range(100):
    action = env.action_space.sample()

    obs, rew, terminated, truncated, info = env.step(action)

    print(action)

env.close()


def get_env(env_cfg):
    return RlMusTermWrapper(
        RlMusRewWrapper(RlMusFlatAct(RlMusFlattenObs(RlMus(env_cfg))))
    )


def run(
    multiagent=DEFAULT_MA,
    output_folder=DEFAULT_OUTPUT_FOLDER,
    gui=DEFAULT_GUI,
    plot=True,
    colab=DEFAULT_COLAB,
    record_video=DEFAULT_RECORD_VIDEO,
    local=True,
):
    """https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/pybullet.ipynb

    Args:
        multiagent (_type_, optional): _description_. Defaults to DEFAULT_MA.
        output_folder (_type_, optional): _description_. Defaults to DEFAULT_OUTPUT_FOLDER.
        gui (_type_, optional): _description_. Defaults to DEFAULT_GUI.
        plot (bool, optional): _description_. Defaults to True.
        colab (_type_, optional): _description_. Defaults to DEFAULT_COLAB.
        record_video (_type_, optional): _description_. Defaults to DEFAULT_RECORD_VIDEO.
        local (bool, optional): _description_. Defaults to True.
    """

    filename = os.path.join(
        output_folder,
        "save-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S") + "_" + get_git_hash(),
    )
    if not os.path.exists(filename):
        os.makedirs(filename + "/")

    train_env = make_vec_env(
        lambda: RlMusTermWrapper(
            RlMusRewWrapper(RlMusFlatAct(RlMusFlattenObs(RlMus(env_cfg))))
        ),
        # get_env,
        env_kwargs=dict(),
        n_envs=16,
        seed=0,
    )

    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    eval_env = make_vec_env(
        lambda: RlMusTermWrapper(
            RlMusRewWrapper(RlMusFlatAct(RlMusFlattenObs(RlMus(env_cfg))))
        ),
        env_kwargs=dict(),
        n_envs=1,
        seed=1,
    )

    eval_env = VecNormalize(
        Monitor(eval_env, None, allow_early_resets=True),
        norm_obs=True,
        norm_reward=True,
    )

    #### Check the environment's spaces ########################
    print("[INFO] Action space:", train_env.action_space)
    print("[INFO] Observation space:", train_env.observation_space)

    #### Train the model #######################################
    model = PPO(
        "MlpPolicy",
        train_env,
        tensorboard_log=filename + "/tb/",
        verbose=1,
    )

    #### Target cumulative rewards (problem-dependent) ##########
    target_reward = 8000
    callback_on_best = StopTrainingOnRewardThreshold(
        reward_threshold=target_reward, verbose=1
    )
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        verbose=1,
        best_model_save_path=filename + "/",
        log_path=filename + "/",
        eval_freq=int(1000),
        # n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    model.learn(
        total_timesteps=(
            int(3e7) if local else int(1e2)
        ),  # shorter training in GitHub Actions pytest
        callback=eval_callback,
        log_interval=100,
    )

    #### Save the model ########################################
    model.save(filename + "/final_model.zip")
    print(filename)
    # saving VecNormalize statistics
    train_env.save(filename + "/vec_normalize.pkl")

    #### Print training progression ############################
    with np.load(filename + "/evaluations.npz") as data:
        for j in range(data["timesteps"].shape[0]):
            print(str(data["timesteps"][j]) + "," + str(data["results"][j][0]))

    ############################################################
    ############################################################
    ############################################################
    ############################################################
    ############################################################


    # if local:
    #     input("Press Enter to continue...")

    # filename = r"/home/prime/Documents/workspace/rl-mus-pybullet/results/save-02.27.2024_01.14.02_4f261a6"
    # if os.path.isfile(filename + "/final_model.zip"):
    #     path = filename + "/final_model.zip"
    # if os.path.isfile(filename + "/best_model.zip"):
    #     path = filename + "/best_model.zip"
    # else:
    #     print("[ERROR]: no model under the specified path", filename)
    # model = PPO.load(path)

    # # #### Show (and record a video of) the model's performance ##
    # # test_env = HoverAviary(gui=gui,
    # #                            obs=DEFAULT_OBS,
    # #                            act=DEFAULT_ACT,
    # #                            record=record_video)
    # # test_env_nogui = RlMusRewWrapper()

    # # logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
    # #             num_drones=DEFAULT_AGENTS if multiagent else 1,
    # #             output_folder=output_folder,
    # #             colab=colab
    # #             )

    # # test_env_nogui = VecNormalize.load()
    # # mean_reward, std_reward = evaluate_policy(model,
    # #                                           test_env_nogui,
    # #                                           n_eval_episodes=10
    # #                                           )
    # # print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")
    # env_cfg = {"num_uavs": 1, "renders": True}
    # # test_env = RlMusTermWrapper()

    # test_env = make_vec_env(
    #     RlMusTermWrapper,
    #     env_kwargs=dict(),
    #     #  env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT),
    #     n_envs=1,
    #     seed=0,
    # )

    # # # Load the saved statistics
    # # env = make_vec_env(
    # #     "HalfCheetahBulletEnv-v0",
    # #     env_kwargs=dict(apply_api_compatibility=True),
    # #     n_envs=1,
    # # )

    # with open(
    #     r"/home/prime/Documents/workspace/rl-mus-pybullet/configs/sim_config.cfg", "rt"
    # ) as f:
    #     config = json.load(f)
    # log_config = config["logger_config"]
    # # env_logger = EnvLogger(num_uavs=1, log_config=log_config)
    # # for uav in env.uavs.values():
    # # env_logger.add_uav(0)
    # stats_path = filename + "/vec_normalize.pkl"
    # test_env = VecNormalize.load(stats_path, test_env)
    # #  do not update them at test time
    # test_env.training = False
    # # reward normalization is not needed at test time
    # test_env.norm_reward = False

    # # obs, info = test_env.reset(seed=42, options={})
    # # obs, info = test_env.reset()
    # obs = test_env.reset()
    # start = time.time()
    # for i in range(1000):
    #     action, _states = model.predict(obs, deterministic=True)
    #     # obs, reward, terminated, truncated, info = test_env.step(action)
    #     obs, reward, terminated, info = test_env.step(action)
    #     # env_logger.log(
    #     #     eps_num=0, info=info, obs={0: obs}, reward={0: reward}, action={0: action}
    #     # )
    #     test_env.render()
    #     # sync(i, start, 1 / test_envenv_freq)
    #     if terminated:
    #         print(
    #             "Obs:",
    #             obs,
    #             "\tAction",
    #             action,
    #             "\tReward:",
    #             reward,
    #             "\tTerminated:",
    #             terminated,
    #             # "\tTruncated:",
    #             # truncated,
    #         )
    #         # env_logger.plot_env()
    #         # env_logger.plot(plt_action=True, plt_target=True)
    #         # obs, info = test_env.reset()
    #         obs = test_env.reset()
    #         # break
    # test_env.close()


if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(
        description="Single agent reinforcement learning example script"
    )
    parser.add_argument(
        "--multiagent",
        default=DEFAULT_MA,
        type=str2bool,
        help="Whether to use example LeaderFollower instead of Hover (default: False)",
        metavar="",
    )
    parser.add_argument(
        "--gui",
        default=DEFAULT_GUI,
        type=str2bool,
        help="Whether to use PyBullet GUI (default: True)",
        metavar="",
    )
    parser.add_argument(
        "--record_video",
        default=DEFAULT_RECORD_VIDEO,
        type=str2bool,
        help="Whether to record a video (default: False)",
        metavar="",
    )
    parser.add_argument(
        "--output_folder",
        default=DEFAULT_OUTPUT_FOLDER,
        type=str,
        help='Folder where to save logs (default: "results")',
        metavar="",
    )
    parser.add_argument(
        "--colab",
        default=DEFAULT_COLAB,
        type=bool,
        help='Whether example is being run by a notebook (default: "False")',
        metavar="",
    )
    ARGS = parser.parse_args()

    run(**vars(ARGS))
