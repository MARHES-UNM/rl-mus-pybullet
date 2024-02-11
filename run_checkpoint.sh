#!/bin/bash 

CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/results/PPO/multi-uav-sim-v0_2024-02-09-18-14_0cee8a8/pen_crashed/pen_crashed/PPO_multi-uav-sim-v0_eebad_00000_0_2024-02-09_18-14-10/checkpoint_000120/policies/shared_policy"
CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/results/PPO/multi-uav-sim-v0_2024-02-10-12-16_0a2fa9c/static_tgts/static_tgts/PPO_multi-uav-sim-v0_2efff_00000_0_stp_penalty=5_2024-02-10_12-16-52/checkpoint_000120/policies/shared_policy"
python run_experiment.py \
    test \
    --checkpoint ${CHECKPOINT} --plot_results --renders --seed None