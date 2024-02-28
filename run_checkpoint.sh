#!/bin/bash 

CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/results/train/PPO/rl-mus-v0_2024-02-24-22-07_ae8b008_1u_0o/mean_filter/mean_filter/PPO_rl-mus-v0_110ac_00000_0_2024-02-24_22-07-53/checkpoint_000325/policies/shared_policy"
CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/results/train/PPO/rl-mus-v0_2024-02-25-07-07_d6c1e8f_1u_0o/randPos/randPos/PPO_rl-mus-v0_65f3c_00000_0_2024-02-25_07-07-08/checkpoint_000453/policies/shared_policy"
CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/results/train/PPO/rl-mus-v0_2024-02-27-18-32_ec9b0dc_1u_0o/norm_obs/norm_obs/PPO_rl-mus-v0_84814_00000_0_2024-02-27_18-32-48/checkpoint_000453/policies/shared_policy"
python run_experiment.py \
    test \
    --checkpoint ${CHECKPOINT} --plot_results --renders --seed "None" "--max_num_episodes" 4