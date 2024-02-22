#!/bin/bash 

CHECKPOINT=""
python run_experiment.py \
    test \
    --checkpoint ${CHECKPOINT} --plot_results --renders --seed "None"