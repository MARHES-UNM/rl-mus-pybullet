#!/bin/bash 

CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/results/train/PPO/rl-mus-v0_2024-02-24-15-04_87f97e3_1u_0o/no_norm_orbs/no_norm_orbs/PPO_rl-mus-v0_dfaa1_00000_0_2024-02-24_15-04-10/checkpoint_000050/policies/shared_policy"
python run_experiment.py \
    test \
    --checkpoint ${CHECKPOINT} --plot_results --renders --seed "None"