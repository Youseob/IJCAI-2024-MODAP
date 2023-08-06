#!/bin/bash
# dataset=hopper-medium-replay
# init_num_model=7
# num_model=5
# H=5
reward_type=penalized_reward
# lam=0.2
# seed=0
# num_worker=4
python required.py

python exp_maple.py --algo_name=maple \
                    --dataset=$dataset \
                    --init_num_model=$init_num_model \
                    --num_model=$num_model \
                    --H=$H \
                    --reward_type=$reward_type \
                    --lam=$lam \
                    --num_worker=$num_worker \
                    --seed=$seed