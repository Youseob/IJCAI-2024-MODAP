#!/bin/bash
# dataset=$1
init_num=110
num_model=100
add_value=True
temp=None
q_lambda=10
belief_mode=bay
dataset=walker2d-random
seed=42
# num_traj_infer=$4
# seed=$5
# WANDB_MODE=disabled python examples/train_d4rl.py --algo_name=bayrl_calib \
#                               --exp_name=$prior_reg-reg-adv_bay_rl-$num_model-ensemble-$belief_mode-$temp-add-value-$add_value \
#                               --task=d4rl-$dataset-v2 \
#                               --transition_init_num=$init_num \
#                               --transition_select_num=$num_model \
#                               --dynamics_path=/root/$dataset-$num_model-seed-$seed.th \
#                               --traj_num_to_infer=$num_traj_infer \
#                               --belief_update_mode=$belief_mode \
#                               --temp=$temp \
#                               --seed=$seed
                            
WANDB_MODE=disabled python examples/train_d4rl.py --algo_name=pessi_bayrl \
                              --exp_name=test \
                              --task=d4rl-$dataset-v2 \
                              --transition_init_num=110 \
                              --transition_select_num=$num_model \
                              --dynamics_path=/root/$dataset-$num_model-seed-$seed.th \
                              --traj_num_to_infer=100 \
                              --belief_update_mode=bay \
                              --temp=$temp \
                              --q_lambda=10 \
                              --add_value_to_rt=True \
                              --seed=$seed