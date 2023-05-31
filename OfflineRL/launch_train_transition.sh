#!/bin/bash
init_num=$1
num_model=$2
seed=$3
export WANDB_API_KEY=46f753c002a9fa94863acc64899744295fe92165 
WANDB_MODE=disabled python examples/train_d4rl.py --algo_name=model_analysis \
                              --exp_name=bay_rl-$num_model-ensemble \
                              --task=d4rl-$dataset-v2 \
                              --transition_init_num=$init_num \
                              --transition_select_num=$num_model \
                              --transition_lr=1e-3 \
                              --only_dynamics=True \
                              --dynamics_save_path=/root/$dataset-$num_model-seed-$seed.th \
                              --seed=$seed \
                              --model_type='esnn'
