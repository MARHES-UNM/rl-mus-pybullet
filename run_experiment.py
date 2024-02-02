import argparse
from datetime import datetime
from time import time
from matplotlib import pyplot as plt
import numpy as np

from rl_mus.envs.rl_mus import RlMus
from pathlib import Path
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from rl_mus.utils.env_utils import get_git_hash
from rl_mus.utils.logger import EnvLogger

import os
import logging
import json
from rl_mus.utils.logger import UavLogger


PATH = Path(__file__).parent.absolute().resolve()
logger = logging.getLogger(__name__)


def setup_stream(logging_level=logging.DEBUG):
    # Turns on logging to console
    ch = logging.StreamHandler()
    ch.setLevel(logging_level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - "
        "<%(module)s:%(funcName)s:%(lineno)s> - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(logging_level)


def experiment(exp_config={}, max_num_episodes=1, experiment_num=0):
    fname = exp_config.setdefault("fname", None)
    write_experiment = exp_config.setdefault("write_experiment", False)
    env_config = exp_config["env_config"]
    plot_results = exp_config["plot_results"]
    log_config = exp_config["logger_config"]

    # Need to create a temporary environment to get obs and action space
    renders = env_config["renders"]
    env_config['renders'] = False
    temp_env = RlMus(env_config)
    env_obs_space = temp_env.observation_space[temp_env.first_uav_id]
    env_action_space = temp_env.action_space[temp_env.first_uav_id]
    temp_env.close()
    env_config['renders'] = renders

    # get the algorithm or policy to run
    algo_to_run = exp_config["exp_config"].setdefault("run", "PPO")
    if algo_to_run not in ["cc", "PPO"]:
        print("Unrecognized algorithm. Exiting...")
        exit(99)

    if algo_to_run == "PPO":
        checkpoint = exp_config["exp_config"].setdefault("checkpoint", None)

        # Reload the algorithm as is from training.
        if checkpoint is not None:
            # use policy here instead of algorithm because it's more efficient
            use_policy = True
            from ray.rllib.policy.policy import Policy
            from ray.rllib.models.preprocessors import get_preprocessor

            algo = Policy.from_checkpoint(checkpoint)

            # need preprocesor here if using policy
            # https://docs.ray.io/en/releases-2.6.3/rllib/rllib-training.html
            prep = get_preprocessor(env_obs_space)(env_obs_space)
        else:
            use_policy = False
            algo = (
                PPOConfig()
                .environment(
                    env=exp_config["env_name"], env_config=exp_config["env_config"]
                )
                .framework("torch")
                .rollouts(num_rollout_workers=0)
                .debugging(log_level="ERROR", seed=exp_config["env_config"]["seed"])
                .resources(
                    num_gpus=0,
                    placement_strategy=[{"cpu": 1}, {"cpu": 1}],
                    num_gpus_per_learner_worker=0,
                )
                .multi_agent(
                    policies={
                        "shared_policy": (
                            None,
                            env_obs_space,
                            env_action_space,
                            {},
                        )
                    },
                    # Always use "shared" policy.
                    policy_mapping_fn=(
                        lambda agent_id, episode, worker, **kwargs: "shared_policy"
                    ),
                )
                .build()
            )
            # restore algorithm if need be:
            # algo.restore(checkpoint)

    # if exp_config["exp_config"]["safe_action_type"] == "nn_cbf":
    #     sl = SafetyLayer(env, exp_config["safety_layer_cfg"])

    env = algo.workers.local_worker().env

    env_logger = EnvLogger(num_uavs=env.num_uavs, log_config=log_config)
    for uav in env.uavs.values():
        env_logger.add_uav(uav.id)

    (obs, info), done = env.reset(), {i.id: False for i in env.uavs.values()}

    done["__all__"] = False

    logger.debug("running experiment")
    num_episodes = 0
    start_time = time()

    time_step = 0
    while num_episodes < max_num_episodes:
        actions = {}
        for uav in env.uavs.values():
            idx = uav.id
            # classic control
            if algo_to_run == "cc":
                actions[idx] = env.get_time_coord_action(env.uavs[idx])
            elif algo_to_run == "PPO":
                if use_policy:
                    actions[idx] = algo.compute_single_action(prep.transform(obs[idx]))[
                        0
                    ]
                else:
                    actions[idx] = algo.compute_single_action(
                        obs[idx], policy_id="shared_policy"
                    )

            if exp_config["exp_config"]["safe_action_type"] is not None:
                if exp_config["exp_config"]["safe_action_type"] == "cbf":
                    actions[idx] = env.get_safe_action(env.uavs[idx], actions[idx])
                elif exp_config["exp_config"]["safe_action_type"] == "nn_cbf":
                    actions[idx] = sl.get_action(obs[idx], actions[idx])
                elif exp_config["exp_config"]["safe_action_type"] == "sca":
                    actions[idx] = env.get_col_avoidance(env.uavs[idx], actions[idx])
                else:
                    print("unknow safe action type")

        obs, rew, done, truncated, info = env.step(actions)
        if time_step % (env.env_freq / env_logger.log_freq) == 0:
            env_logger.log(
                eps_num=num_episodes, info=info, obs=obs, reward=rew, action=actions
            )

        if renders:
            env.render()

        if done["__all__"]:
            num_episodes += 1

            if plot_results:
                env_logger.plot(plt_action=True)

            if num_episodes == max_num_episodes:
                end_time = time() - start_time
                break

            (obs, info), done = env.reset(), {
                agent.id: False for agent in env.uavs.values()
            }
            done["__all__"] = False

        time_step += 1

    env.close()

    # TODO: set log time
    # env_logger.log_total_time()
    # TODO: fix logging the experiment
    # if write_experiment:
    #     if fname is None:
    #         file_prefix = {
    #             "tgt_v": env_config["target_v"],
    #             "sa": env_config["use_safe_action"],
    #             "obs": env_config["num_obstacles"],
    #             "seed": env_config["seed"],
    #         }
    #         file_prefix = "_".join(
    #             [f"{k}_{str(int(v))}" for k, v in file_prefix.items()]
    #         )

    #         fname = f"exp_{experiment_num}_{file_prefix}_result.json"
    #     # writing too much data, for now just save the first experiment
    #     for k, v in results["episode_data"].items():
    #         results["episode_data"][k] = [
    #             v[0],
    #         ]

    #     results["env_config"] = env.env_config
    #     results["exp_config"] = exp_config["exp_config"]
    #     results["time_total_s"] = end_time
    #     with open(fname, "w") as f:
    #         json.dump(results, f)

    logger.debug("done")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_config", default=f"{PATH}/configs/sim_config.cfg")
    parser.add_argument(
        "--log_dir",
    )
    parser.add_argument("-d", "--debug")
    parser.add_argument("-v", help="version number of experiment")
    parser.add_argument("--max_num_episodes", type=int, default=1)
    parser.add_argument("--experiment_num", type=int, default=0)
    parser.add_argument("--env_name", type=str, default="multi-uav-sim-v0")
    parser.add_argument("--renders", action="store_true", default=False)
    parser.add_argument("--write_exp", action="store_true")
    parser.add_argument("--plot_results", action="store_true", default=False)

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    setup_stream()

    with open(args.load_config, "rt") as f:
        args.config = json.load(f)

    if not args.config["exp_config"]["run"] == "cc":
        if args.env_name == "multi-uav-sim-v0":
            args.config["env_name"] = args.env_name
            tune.register_env(
                args.config["env_name"],
                lambda env_config: RlMus(env_config=env_config),
            )

    logger.debug(f"config: {args.config}")

    if not args.log_dir:
        branch_hash = get_git_hash()

        dir_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        args.log_dir = f"./results/test_results/exp_{dir_timestamp}_{branch_hash}"

    max_num_episodes = args.max_num_episodes
    experiment_num = args.experiment_num
    args.config["env_config"]["renders"] = args.renders
    args.config["plot_results"] = args.plot_results

    if args.write_exp:
        args.config["write_experiment"] = True

        output_folder = Path(args.log_dir)
        if not output_folder.exists():
            output_folder.mkdir(parents=True, exist_ok=True)

        args.config["fname"] = output_folder / "result.json"

    experiment(args.config, max_num_episodes, experiment_num)


if __name__ == "__main__":
    main()
