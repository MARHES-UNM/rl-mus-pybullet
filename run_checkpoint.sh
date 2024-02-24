#!/bin/bash 

CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/results/train/PPO/rl-mus-v0_2024-02-24-12-44_d92e765_1u_0o/no_norm_orbs/no_norm_orbs/PPO_rl-mus-v0_6f15c_00000_0_2024-02-24_12-45-01/checkpoint_000450/policies/shared_policy"
python run_experiment.py \
    test \
    --checkpoint ${CHECKPOINT} --plot_results --renders --seed "None"