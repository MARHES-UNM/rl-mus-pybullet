#!/bin/bash 

CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/results/train/PPO/rl-mus-v0_2024-02-24-22-07_ae8b008_1u_0o/mean_filter/mean_filter/PPO_rl-mus-v0_110ac_00000_0_2024-02-24_22-07-53/checkpoint_000325/policies/shared_policy"
python run_experiment.py \
    test \
    --checkpoint ${CHECKPOINT} --plot_results --renders --seed "None"