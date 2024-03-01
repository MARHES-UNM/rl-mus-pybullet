#!/bin/bash


# python multi_agent_shared_parameter.py --name low_stp --stop-timesteps 20000000
python run_experiment.py --env_name "rl-mus-ttc-v0" --name doe_param --stop-timesteps 20000000
python run_experiment.py --env_name "rl-mus-v0" --name doe_param_base --stop-timesteps 20000000