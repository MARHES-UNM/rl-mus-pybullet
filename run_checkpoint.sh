#!/bin/bash 

CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/results/PPO/multi-uav-sim-v0_2024-02-09-18-14_0cee8a8/pen_crashed/pen_crashed/PPO_multi-uav-sim-v0_eebad_00000_0_2024-02-09_18-14-10/checkpoint_000120/policies/shared_policy"
CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/results/PPO/multi-uav-sim-v0_2024-02-10-12-16_0a2fa9c/static_tgts/static_tgts/PPO_multi-uav-sim-v0_2efff_00000_0_stp_penalty=5_2024-02-10_12-16-52/checkpoint_000120/policies/shared_policy"
CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/results/train/PPO/rl-mus-v0_2024-02-14-22-19_900ca42_1u_0o/beta_5/beta_5/PPO_rl-mus-v0_11b9a_00000_0_beta=5_2024-02-14_22-19-46/checkpoint_000155/policies/shared_policy"
CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/results/train/PPO/rl-mus-v0_2024-02-14-22-19_900ca42_1u_0o/beta_5/beta_5/PPO_rl-mus-v0_11b9a_00000_0_beta=5_2024-02-14_22-19-46/checkpoint_000355/policies/shared_policy"
CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/results/train/PPO/rl-mus-v0_2024-02-16-19-05_e0a49e8_1u_0o/sim_rew/sim_rew/PPO_rl-mus-v0_447d0_00000_0_2024-02-16_19-05-32/checkpoint_000100/policies/shared_policy"
CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/results/train/PPO/rl-mus-v0_2024-02-18-10-49_23c9d6d_1u_0o/static_agent/static_agent/PPO_rl-mus-v0_56ee0_00000_0_2024-02-18_10-49-46/checkpoint_000325/policies/shared_policy"
python run_experiment.py \
    test \
    --checkpoint ${CHECKPOINT} --plot_results --renders --seed "None"