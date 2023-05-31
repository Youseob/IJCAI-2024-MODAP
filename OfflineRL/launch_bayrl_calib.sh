#!/bin/bash
# dataset=$1
init_num=$1
num_model=$2
# num_traj_infer=$4
# seed=$5
WANDB_MODE=disabled python examples/train_d4rl.py --algo_name=model_analysis \
                              --exp_name=bay_rl-$num_model-ensemble-calibration \
                              --task=d4rl-$dataset-v2 \
                              --transition_init_num=$init_num \
                              --transition_select_num=$num_model \
                              --dynamics_path=/root/$dataset-$num_model-seed-$seed.th \
                              --traj_num_to_infer=$num_traj_infer \
                              --belief_update_mode=bay \
                              --calibration=True \
                              --seed=$seed  