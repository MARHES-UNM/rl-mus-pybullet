#!/bin/bash 

CHECKPOINT="checkpoints/checkpoint_000301/policies/shared_policy"

python run_experiment.py \
    --env_name rl-mus-ttc-v0 \
    test \
    --checkpoint ${CHECKPOINT} --plot_results --renders --seed "None" "--max_num_episodes" 1
