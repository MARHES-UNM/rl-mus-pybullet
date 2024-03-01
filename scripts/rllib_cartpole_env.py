#%% 
# import gymnasium as gym
import gymnasium as gym

import pybullet_envs
from ray import tune
# from ray.rllib.algorithms.ppo import
from ray.tune.registry import register_env, get_trainable_cls
from ray import air
from gymnasium.wrappers import EnvCompatibility

# %%

# env_name = "CartPoleContinuousBulletEnv-v0"
env_name = "LunarLander-v2"
env_name = 'CartPole-v1'
# env = gym.make("LunarLander-v2", render_mode="human")

def make_env(env_config):
    import pybullet_envs
    return gym.make(env_name)

register_env(env_name, make_env)
algo_config = (
        get_trainable_cls("PPO")
        .get_default_config()
        .environment(env=env_name)
        .framework("torch")
        .rollouts(num_rollout_workers=0)
        .debugging(log_level="ERROR", seed=123)
        .rollouts(
            num_rollout_workers=(
            8
            ),  # set 0 to main worker run sim
            num_envs_per_worker=12,
            # create_env_on_local_worker=True,
            # rollout_fragment_length="auto",
            batch_mode="complete_episodes",
        )
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(
            num_gpus=1,
            num_learner_workers=1,
            num_gpus_per_learner_worker=1,
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

stop = {
        "training_iteration": 10000,
        "timesteps_total": 5000000,
    }

    # # # trainable_with_resources = tune.with_resources(args.run, {"cpu": 18, "gpu": 1.0})
    # # # If you have 4 CPUs and 1 GPU on your machine, this will run 1 trial at a time.
    # # trainable_with_cpu_gpu = tune.with_resources(algo, {"cpu": 2, "gpu": 1})
tuner = tune.Tuner(
        "PPO",
        # trainable_with_cpu_gpu,
        param_space=algo_config.to_dict(),
        # tune_config=tune.TuneConfig(num_samples=10),
        run_config=air.RunConfig(
            stop=stop,
            local_dir=r"/home/prime/Documents/workspace/rl-mus-pybullet/results",
            name='debug',
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=100,
                # checkpoint_score_attribute="",
                checkpoint_at_end=True,
                checkpoint_frequency=25,
            ),
        ),
    )

results = tuner.fit()
