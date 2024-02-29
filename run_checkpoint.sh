#!/bin/bash 

CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/results/train/PPO/rl-mus-v0_2024-02-24-22-07_ae8b008_1u_0o/mean_filter/mean_filter/PPO_rl-mus-v0_110ac_00000_0_2024-02-24_22-07-53/checkpoint_000325/policies/shared_policy"
CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/results/train/PPO/rl-mus-v0_2024-02-25-07-07_d6c1e8f_1u_0o/randPos/randPos/PPO_rl-mus-v0_65f3c_00000_0_2024-02-25_07-07-08/checkpoint_000453/policies/shared_policy"
CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/results/train/PPO/rl-mus-v0_2024-02-27-23-45_4a85848_1u_0o/rllib_no_filter/rllib_no_filter/PPO_rl-mus-v0_3e504_00000_0_2024-02-27_23-45-48/checkpoint_000185/policies/shared_policy"
CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/results/train/PPO/rl-mus-v0_2024-02-28-00-36_837e744_4u_0o/rllib_no_filter4uav/rllib_no_filter4uav/PPO_rl-mus-v0_5899f_00000_0_2024-02-28_00-36-39/checkpoint_000155/policies/shared_policy"
CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/results/train/PPO/rl-mus-v0_2024-02-27-23-45_4a85848_1u_0o/rllib_no_filter/rllib_no_filter/PPO_rl-mus-v0_3e504_00000_0_2024-02-27_23-45-48/checkpoint_000225/policies/shared_policy"
CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/results/train/PPO/rl-mus-v0_2024-02-28-22-19_7b15a57_1u_0o/pos_rew/pos_rew/PPO_rl-mus-v0_5e09a_00000_0_2024-02-28_22-19-37/checkpoint_000453/policies/shared_policy"
python run_experiment.py \
    test \
    --checkpoint ${CHECKPOINT} --plot_results --renders --seed "None" "--max_num_episodes" 4