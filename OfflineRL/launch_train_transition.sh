#!/bin/bash
dataset=$1
init_num=$2
num_model=$3
seed=$4
export WANDB_API_KEY=46f753c002a9fa94863acc64899744295fe92165 
WANDB_MODE=disabled python examples/train_d4rl.py --algo_name=bayrl \
                              --exp_name=bay_rl-$num_model-ensemble \
                              --task=d4rl-$dataset-v2 \
                              --transition_init_num=$init_num \
                              --transition_select_num=$num_model \
                              --transition_lr=1e-4 \
                              --only_dynamics=True \
                              --dynamics_save_path=/output/$dataset-$num_model-seed-$seed.th \
                              --seed=$seed
