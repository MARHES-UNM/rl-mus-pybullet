#!/bin/bash 

CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/results/train/PPO/rl-mus-v0_2024-02-24-09-31_e08b34c_1u_0o/norm_orbs/norm_orbs/PPO_rl-mus-v0_5cc2d_00000_0_2024-02-24_09-31-13/checkpoint_000025/policies/shared_policy"
python run_experiment.py \
    test \
    --checkpoint ${CHECKPOINT} --plot_results --renders --seed "None"