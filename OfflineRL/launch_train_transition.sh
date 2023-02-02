#!/bin/bash
dataset=$1
init_num=$2
num_model=$3
seed=$4
export WANDB_API_KEY=46f753c002a9fa94863acc64899744295fe92165
export MUJOCO_PY_MUJOCO_PATH=/home/.mujoco/mujoco210
WANDB_MODE=disabled python examples/train_d4rl.py --algo_name=bayrl \
                                                  --exp_name=train_transition \
                                                  --task=d4rl-$dataset-v2 \
                                                  --transition_init_num=$init_num \
                                                  --transition_select_num=$num_model \
                                                  --transition_lr=1e-4 \
                                                  --dynamics_save_path=/root/output/$dataset-$num_model-seed-$seed.th \
                                                  --only_dynamics=True \
                                                  --seed=$seed
