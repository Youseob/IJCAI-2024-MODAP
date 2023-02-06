#!/bin/bash
dataset=$1
init_num=$2
num_model=$3
num_traj_infer=$4
seed=$5
python examples/train_d4rl.py --algo_name=bayrl \
                              --exp_name=bay_rl-$num_model-ensemble \
                              --task=d4rl-$dataset-v2 \
                              --transition_init_num=$init_num \
                              --transition_select_num=$num_model \
                              --dynamics_path=/model/$dataset-$num_model-seed-$seed.th \
                              --traj_num_to_infer=$num_traj_infer \  
                              --seed=$seed
