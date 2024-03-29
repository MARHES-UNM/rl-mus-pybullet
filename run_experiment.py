import argparse
from datetime import datetime
from time import time
from matplotlib import pyplot as plt
import numpy as np

from rl_mus.envs.rl_mus import RlMus
from pathlib import Path
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from rl_mus.envs.rl_mus_ttc import RlMusTtc
from rl_mus.utils.env_utils import get_git_hash
from rl_mus.utils.logger import EnvLogger
from ray.tune.registry import get_trainable_cls
from ray import air, tune
from rl_mus.utils.callbacks import TrainCallback
from ray.rllib.algorithms.callbacks import make_multi_callbacks
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.policy import local_policy_inference

import ray

import os
import logging
import json

PATH = Path(__file__).parent.absolute().resolve()
RESULTS_DIR = Path.home() / "ray_results"
logger = logging.getLogger(__name__)
max_num_cpus = os.cpu_count() - 1


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


def get_obs_act_space(config):

    # Need to create a temporary environment to get obs and action space
    env_config = config["env_config"]
    renders = env_config["renders"]
    env_config["renders"] = False
    if config["env_name"] == "rl-mus-v0":
        temp_env = RlMus(env_config)
    else:
        temp_env = RlMusTtc(env_config)

    env_obs_space = temp_env.observation_space[0]
    env_action_space = temp_env.action_space[0]
    temp_env.close()
    env_config["renders"] = renders

    return env_obs_space, env_action_space


def get_algo_config(config, env_obs_space, env_action_space):

    algo_config = (
        get_trainable_cls(config["exp_config"]["run"])
        .get_default_config()
        .environment(env=config["env_name"], env_config=config["env_config"])
        .framework(config["exp_config"]["framework"])
        .rollouts(
            num_rollout_workers=0,
            # observation_filter="MeanStdFilter",  # or "NoFilter"
        )
        .debugging(log_level="ERROR", seed=config["env_config"]["seed"])
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
    )

    return algo_config


def train(args):

    # args.local_mode = True
    ray.init(local_mode=args.local_mode, num_gpus=1)

    # We get the spaces here before test vary the experiment treatments (factors)
    env_obs_space, env_action_space = get_obs_act_space(args.config)

    # Vary treatments here
    num_gpus = int(os.environ.get("RLLIB_NUM_GPUS", args.gpu))
    args.config["env_config"]["num_uavs"] = tune.grid_search([4])
    args.config["env_config"]["d_thresh"] = tune.grid_search([0.1, 0.01])
    args.config["env_config"]["target_pos_rand"] = True
    # args.config["env_config"]["use_safe_action"] = tune.grid_search([False])

    args.config["env_config"]["tgt_reward"] = 10
    args.config["env_config"]["stp_penalty"] = 0.1
    args.config["env_config"]["time_final"] = 8
    args.config["env_config"]["t_go_max"] = 2.0

    args.config["env_config"]["beta"] = 0.3
    args.config["env_config"]["beta_vel"] = 0.0
    args.config["env_config"]["crash_penalty"] = 1.0
    args.config["env_config"]["uav_collision_weight"] = 0.0

    obs_filter = "NoFilter"
    callback_list = [TrainCallback]
    # multi_callbacks = make_multi_callbacks(callback_list)
    # Common config params: https://docs.ray.io/en/latest/rllib/rllib-training.html#configuring-rllib-algorithms
    train_config = (
        get_algo_config(args.config, env_obs_space, env_action_space).rollouts(
            num_rollout_workers=(
                1 if args.smoke_test else args.num_rollout_workers
            ),  # set 0 to main worker run sim
            num_envs_per_worker=args.num_envs_per_worker,
            # create_env_on_local_worker=True,
            # rollout_fragment_length="auto",
            batch_mode="complete_episodes",
            observation_filter=obs_filter,  # or "NoFilter"
        )
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(
            num_gpus=0 if args.smoke_test else num_gpus,
            num_learner_workers=1,
            num_gpus_per_learner_worker=0 if args.smoke_test else args.gpu,
        )
        # See for changing model options https://docs.ray.io/en/latest/rllib/rllib-models.html
        # .model()
        # See for specific ppo config: https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ppo
        # See for more on PPO hyperparameters: https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
        .training(
            # https://docs.ray.io/en/latest/rllib/rllib-models.html
            # model={"fcnet_hiddens": [512, 512, 512]},
            lr=5e-5,
            use_gae=True,
            use_critic=True,
            lambda_=0.95,
            train_batch_size=65536,
            gamma=0.99,
            num_sgd_iter=32,
            sgd_minibatch_size=4096,
            vf_clip_param=10.0,
            vf_loss_coeff=0.5,
            clip_param=0.2,
            grad_clip=1.0,
            # entropy_coeff=0.0,
            # # seeing if this solves the error:
            # # https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem
            # # Expected parameter loc (Tensor of shape (4096, 3)) of distribution Normal(loc: torch.Size([4096, 3]), scale: torch.Size([4096, 3])) to satisfy the constraint Real(),
            # kl_coeff=1.0,
            # kl_target=0.0068,
        )
        # .reporting(keep_per_episode_custom_metrics=True)
        # .evaluation(
        #     evaluation_interval=10, evaluation_duration=10  # default number of episodes
        # )
    )

    multi_callbacks = make_multi_callbacks(callback_list)
    train_config.callbacks(multi_callbacks)

    stop = {
        "training_iteration": 1 if args.smoke_test else args.stop_iters,
        "timesteps_total": args.stop_timesteps,
    }

    # # # trainable_with_resources = tune.with_resources(args.run, {"cpu": 18, "gpu": 1.0})
    # # # If you have 4 CPUs and 1 GPU on your machine, this will run 1 trial at a time.
    # # trainable_with_cpu_gpu = tune.with_resources(algo, {"cpu": 2, "gpu": 1})
    tuner = tune.Tuner(
        args.run,
        # trainable_with_cpu_gpu,
        param_space=train_config.to_dict(),
        # tune_config=tune.TuneConfig(num_samples=10),
        run_config=air.RunConfig(
            stop=stop,
            local_dir=args.log_dir,
            name=args.name,
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=100,
                # checkpoint_score_attribute="",
                checkpoint_at_end=True,
                checkpoint_frequency=5,
            ),
        ),
    )

    results = tuner.fit()


def test(args):
    if args.tune_run:
        pass
    else:
        experiment(args)


def experiment(args):
    if args.seed:
        print(f"******************seed: {args.seed}")
        seed_val = none_or_int(args.seed)
        args.config["env_config"]["seed"] = seed_val

    args.config["env_config"]["renders"] = args.renders
    args.config["plot_results"] = args.plot_results

    experiment_num = args.experiment_num
    exp_config = args.config

    if args.checkpoint:
        exp_config["exp_config"]["checkpoint"] = args.checkpoint

    max_num_episodes = args.max_num_episodes

    env_config = exp_config["env_config"]
    plot_results = exp_config["plot_results"]
    log_config = exp_config["logger_config"]
    renders = env_config["renders"]

    # get the algorithm or policy to run
    algo_to_run = exp_config["exp_config"].setdefault("run", "PPO")
    if algo_to_run not in ["cc", "PPO"]:
        print("Unrecognized algorithm. Exiting...")
        exit(99)

    if args.config["env_name"] == "rl-mus-v0":
        env = RlMus(env_config)
    else:
        env = RlMusTtc(env_config)

    if algo_to_run == "PPO":
        checkpoint = exp_config["exp_config"].setdefault("checkpoint", None)
        env_obs_space, env_action_space = get_obs_act_space(args.config)

        # Reload the algorithm as is from training.
        if checkpoint is not None:
            # Can use algorithm from checkpoint instead
            # algo = Algorithm.from_checkpoint(checkpoint)
            # policy = algo.workers.local_worker().policy_map["shared_policy"]
            use_policy = True
            # use policy here instead of algorithm because it's more efficient
            from ray.rllib.policy.policy import Policy
            from ray.rllib.models.preprocessors import get_preprocessor

            algo = Policy.from_checkpoint(checkpoint)

            # # need preprocesor here if using policy
            # # https://docs.ray.io/en/releases-2.6.3/rllib/rllib-training.html
            prep = get_preprocessor(env_obs_space)(env_obs_space)

            # # https://github.com/ray-project/ray/issues/37974#issuecomment-1766994593
            # # TODO: look into issue not being able to use meanstdfilter during evaluation
            # c = list(
            #     filter(
            #         lambda x: type(x)
            #         == ray.rllib.connectors.agent.mean_std_filter.MeanStdObservationFilterAgentConnector,
            #         # algo.workers.local_worker()
            #         # .policy_map["shared_policy"]
            #         algo.agent_connectors.connectors,
            #     )
            # )[0]
        else:
            use_policy = False
            env.close()
            algo = (
                get_algo_config(exp_config, env_obs_space, env_action_space)
                .resources(
                    num_gpus=0,
                    placement_strategy=[{"cpu": 1}, {"cpu": 1}],
                    num_gpus_per_learner_worker=0,
                )
                .build()
            )

            env = algo.workers.local_worker().env

    env_logger = EnvLogger(num_uavs=env.num_uavs, log_config=log_config)

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
            # TODO: fix this controller implementation here
            # classic control
            if algo_to_run == "cc":
                action = np.zeros(4)
                acc  = env.get_time_coord_action(env.uavs[idx])
                cur_vel = env.uavs[idx].vel

                action[:3] = cur_vel + env.sim_dt * acc
                action[3] = np.linalg.norm(action[:3])
                actions[idx] = action
            elif algo_to_run == "PPO":
                if use_policy:

                    # https://github.com/ray-project/ray/issues/41382
                    # # Use local_policy_inference() to run inference, so we do not have to
                    # # provide policy states or extra fetch dictionaries.
                    # # "env_1" and "agent_1" are dummy env and agent IDs to run connectors with.
                    # policy_outputs = local_policy_inference(
                    #     policy, "env_1", "agent_1", obs[idx], #explore=False
                    # )
                    # assert len(policy_outputs) == 1
                    # action, _, _ = policy_outputs[0]
                    # print(f"step {time_step}", obs, action)

                    # actions[idx] = action
                    # actions[idx] = algo.compute_single_action(
                    # c.filter(prep.transform(obs[idx])),  # clip_actions=True
                    # c.filter(obs[idx]),
                    # obs[idx],
                    # policy_id="shared_policy",  # clip_actions=True
                    # )
                    # c = policy.agent_connectors.connectors[1]
                    c = algo.agent_connectors.connectors[1]
                    actions[idx] = algo.compute_single_action(
                        # c.filter(prep.transform(obs[idx])),
                        prep.transform(obs[idx]),
                        clip_actions=True,
                    )[0]
                else:
                    actions[idx] = algo.compute_single_action(
                        obs[idx], policy_id="shared_policy", clip_actions=True
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
        # if time_step % (env.env_freq / env_logger.log_freq) == 0:
        env_logger.log(
            eps_num=num_episodes, info=info, obs=obs, reward=rew, action=actions
        )

        if renders:
            env.render()

        if done["__all__"] or truncated["__all__"]:

            end_time = time() - start_time
            logger.info(f"time_elapsed: {env.time_elapsed}")
            env_logger.log_eps_time(sim_time=env.time_elapsed, real_time=end_time)
            num_episodes += 1
            if num_episodes == max_num_episodes:
                break

            # reset the episode data
            start_time = time()

            (obs, info), done = env.reset(), {
                agent.id: False for agent in env.uavs.values()
            }
            done["__all__"] = False

        time_step += 1

    if plot_results:
        env_logger.plot_env()
        env_logger.plot(plt_action=True, plt_target=True)

    env.close()

    if args.write_exp:
        output_folder = Path(args.log_dir)
        if not output_folder.exists():
            output_folder.mkdir(parents=True, exist_ok=True)

        fname = output_folder / "result.json"

        results = env_logger.data
        results["env_config"] = env.env_config
        results["exp_config"] = args.config["exp_config"]
        with open(fname, "w") as f:
            json.dump(results, f)

    logger.debug("done")


def none_or_int(value):
    if value == "None":
        return None
    return int(value)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_config", default=f"{PATH}/configs/sim_config.cfg")
    parser.add_argument(
        "--log_dir",
    )
    parser.add_argument(
        "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
    )
    parser.add_argument("--name", help="Name of experiment.", default="debug")
    parser.add_argument(
        "--framework",
        choices=["tf", "tf2", "torch"],
        default="torch",
        help="The DL framework specifier.",
    )
    parser.add_argument("--env_name", type=str, default="rl-mus-v0")

    subparsers = parser.add_subparsers(dest="command")
    test_sub = subparsers.add_parser("test")
    test_sub.add_argument("-d", "--debug")
    test_sub.add_argument("--checkpoint")
    test_sub.add_argument("-v", help="version number of experiment")
    test_sub.add_argument("--max_num_episodes", type=int, default=1)
    test_sub.add_argument("--experiment_num", type=int, default=0)
    test_sub.add_argument("--renders", action="store_true", default=False)
    test_sub.add_argument("--write_exp", action="store_true", default=False)
    test_sub.add_argument("--plot_results", action="store_true", default=False)
    test_sub.add_argument("--tune_run", action="store_true", default=False)
    test_sub.add_argument("--seed")

    test_sub.set_defaults(func=test)

    train_sub = subparsers.add_parser("train")
    train_sub.add_argument("--smoke_test", action="store_true", help="run quicktest")

    train_sub.add_argument(
        "--stop_iters", type=int, default=1000, help="Number of iterations to train."
    )
    train_sub.add_argument(
        "--stop_timesteps",
        type=int,
        default=int(30e6),
        help="Number of timesteps to train.",
    )

    train_sub.add_argument(
        "--local-mode",
        action="store_true",
        help="Init Ray in local mode for easier debugging.",
    )

    train_sub.add_argument("--checkpoint", type=str)
    train_sub.add_argument("--cpu", type=int, default=8)
    train_sub.add_argument("--gpu", type=int, default=0.50)
    train_sub.add_argument("--num_envs_per_worker", type=int, default=12)
    train_sub.add_argument("--num_rollout_workers", type=int, default=8)
    train_sub.set_defaults(func=train)

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    setup_stream()

    with open(args.load_config, "rt") as f:
        args.config = json.load(f)

    args.config["env_name"] = args.env_name

    logger.debug(f"config: {args.config}")

    if not args.log_dir:
        branch_hash = get_git_hash()

        num_uavs = args.config["env_config"]["num_uavs"]
        num_obs = args.config["env_config"]["max_num_obstacles"]
        dir_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        log_dir = f"{args.func.__name__}/{args.run}/{args.env_name}_{dir_timestamp}_{branch_hash}_{num_uavs}u_{num_obs}o/{args.name}"
        args.log_dir = RESULTS_DIR / log_dir

    args.log_dir = Path(args.log_dir).resolve()
    # TODO: create log dir when running experiments
    # if not args.log_dir.exists():
    #     args.log_dir.mkdir(parents=True, exist_ok=True)

    args.func(args)


if __name__ == "__main__":
    main()
