#!/bin/bash 

CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/results/train/PPO/rl-mus-v0_2024-02-24-22-07_ae8b008_1u_0o/mean_filter/mean_filter/PPO_rl-mus-v0_110ac_00000_0_2024-02-24_22-07-53/checkpoint_000325/policies/shared_policy"
CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/results/train/PPO/rl-mus-v0_2024-02-25-07-07_d6c1e8f_1u_0o/randPos/randPos/PPO_rl-mus-v0_65f3c_00000_0_2024-02-25_07-07-08/checkpoint_000453/policies/shared_policy"
CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/results/train/PPO/rl-mus-v0_2024-02-27-23-45_4a85848_1u_0o/rllib_no_filter/rllib_no_filter/PPO_rl-mus-v0_3e504_00000_0_2024-02-27_23-45-48/checkpoint_000185/policies/shared_policy"
CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/results/train/PPO/rl-mus-v0_2024-02-28-00-36_837e744_4u_0o/rllib_no_filter4uav/rllib_no_filter4uav/PPO_rl-mus-v0_5899f_00000_0_2024-02-28_00-36-39/checkpoint_000155/policies/shared_policy"
CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/results/train/PPO/rl-mus-v0_2024-02-27-23-45_4a85848_1u_0o/rllib_no_filter/rllib_no_filter/PPO_rl-mus-v0_3e504_00000_0_2024-02-27_23-45-48/checkpoint_000225/policies/shared_policy"
CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/results/train/PPO/rl-mus-v0_2024-02-28-22-19_7b15a57_1u_0o/pos_rew/pos_rew/PPO_rl-mus-v0_5e09a_00000_0_2024-02-28_22-19-37/checkpoint_000453/policies/shared_policy"
CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/results/train/PPO/rl-mus-v0_2024-02-29-18-18_094c162_1u_0o/crash_penalty/crash_penalty/PPO_rl-mus-v0_cc71a_00001_1_crash_penalty=10_2024-02-29_18-18-08/checkpoint_000452/policies/shared_policy"
CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/results/train/PPO/rl-mus-v0_2024-02-29-22-06_9d2a8e7_1u_0o/crash_penalty/crash_penalty/PPO_rl-mus-v0_ab972_00001_1_crash_penalty=10_2024-02-29_22-06-17/checkpoint_000315/policies/shared_policy"
CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/results/train/PPO/rl-mus-v0_2024-03-01-01-08_417bfe8_4u_0o/doe_n_uavs/doe_n_uavs/PPO_rl-mus-v0_19bcd_00002_2_crash_penalty=10,num_uavs=1,target_pos_rand=True,observation_filter=NoFilter_2024-03-01_01-08-19/checkpoint_000452/policies/shared_policy"
CHECKPOINT="/home/prime/ray_results/train/PPO/rl-mus-v0_2024-03-02-07-51_ea435c5_4u_0o/doe_param_base/doe_param_base/PPO_rl-mus-v0_93bd1_00001_1_crash_penalty=10,d_thresh=0.0001,num_uavs=4,stp_penalty=5,time_final=10_2024-03-02_07-51-24/checkpoint_000080/policies/shared_policy"
# CHECKPOINT='/home/prime/Documents/workspace/rl-mus-pybullet/ray_results/rl-mus-v0_2024-03-01-22-10_0ad4abc_4u_0o/doe_param_base/doe_param_base/PPO_rl-mus-v0_66aea_00000_0_crash_penalty=10,d_thresh=0.0001,num_uavs=1,stp_penalty=5,time_final=10_2024-03-01_22-10-19/checkpoint_000297/policies/shared_policy'
CHECKPOINT='/home/prime/Documents/workspace/rl-mus-pybullet/ray_results/rl-mus-v0_2024-03-01-01-08_417bfe8_4u_0o/doe_n_uavs/doe_n_uavs/PPO_rl-mus-v0_19bcd_00002_2_crash_penalty=10,num_uavs=1,target_pos_rand=True,observation_filter=NoFilter_2024-03-01_01-08-19/checkpoint_000452/policies/shared_policy'
CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/ray_results/train/PPO/rl-mus-v0_2024-03-02-13-21_5103c92_1u_0o/rlmus_baseline/rlmus_baseline/PPO_rl-mus-v0_c2560_00000_0_crash_penalty=10,num_uavs=1_2024-03-02_13-21-59/checkpoint_000302/policies/shared_policy" 
CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/ray_results/train/PPO/rl-mus-ttc-v0_2024-03-03-00-37_5103c92_1u_0o/ttc_baseline/ttc_baseline/PPO_rl-mus-ttc-v0_1a5d0_00000_0_crash_penalty=10,d_thresh=0.1500,num_uavs=1,stp_penalty=5,time_final=20_2024-03-03_00-37-19/checkpoint_000295/policies/shared_policy"
CHECKPOINT="/home/prime/Documents/workspace/rl-mus-pybullet/ray_results/train/PPO/rl-mus-ttc-v0_2024-03-05-17-27_a120051_4u_0o/n_uavs_params_train/n_uavs_params_train/PPO_rl-mus-ttc-v0_9669b_00000_0_beta=0.3000,beta_vel=0.0000,crash_penalty=1.0000,d_thresh=0.1500,num_uavs=4,stp_penalty=0.1000,t_g_2024-03-05_17-27-44/checkpoint_000301/policies/shared_policy"

python run_experiment.py \
    --env_name rl-mus-ttc-v0 \
    test \
    --checkpoint ${CHECKPOINT} --plot_results --renders --seed "None" "--max_num_episodes" 1
