#!/bin/bash
init_num=110
num_model=100
add_value=True
temp=None
q_lambda=10
belief_mode=bay
dataset=walker2d-random
seed=42
WANDB_MODE=disabled python examples/train_d4rl.py --algo_name=bayrl_v2 \
                              --exp_name=test \
                              --task=d4rl-$dataset-v2 \
                              --transition_init_num=110 \
                              --transition_select_num=$num_model \
                              --dynamics_path=/root/$dataset-$num_model-seed-$seed.th \
                              --traj_num_to_infer=50 \
                              --belief_update_mode=bay \
                              --temp=$temp \
                              --q_lambda=10 \
                            #   --add_value_to_rt=True \
                              --seed=$seed