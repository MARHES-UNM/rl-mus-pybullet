#!/bin/bash 

CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/results/train/PPO/rl-mus-v0_2024-02-24-15-23_2caa193_1u_0o/init_vals/init_vals/PPO_rl-mus-v0_95868_00000_0_2024-02-24_15-23-34/checkpoint_000175/policies/shared_policy"
python run_experiment.py \
    test \
    --checkpoint ${CHECKPOINT} --plot_results --renders --seed "None"