#!/bin/bash
# dataset=walker2d-random
num_model=100
# num_traj_infer=100
# add_value=True
cal=True
r_scale=10
belief_mode=bay
dataset=walker2d-random
seed=42
horizon_step=5

WANDB_MODE=disabled python examples/train_d4rl.py --algo_name=adv_bayrl_v3 \
                              --exp_name=adv-bay-r_scale-$r_scale \
                              --task=d4rl-$dataset-v2 \
                              --transition_init_num=110 \
                              --transition_select_num=$num_model \
                              --dynamics_path=/root/$dataset-$num_model-seed-$seed.th \
                              --traj_num_to_infer=$num_traj_infer \
                              --belief_update_mode=$belief_mode \
                              --add_value_to_rt=$add_value \
                              --b_lr=1e-4 \
                              --horizon=$horizon_step
                              --reward_scale=$r_scale \
                              --calibration=$cal \
                              --seed=$seed