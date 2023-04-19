#!/bin/bash
# dataset=walker2d-random
num_model=100
# num_traj_infer=100
add_value=True
cal=True
# r_scale=10
belief_mode=bay
# seed=42
horizon_step=5

python examples/train_d4rl.py --algo_name=pessi_bayrl \
                              --exp_name=adv-$lambda-add_value-$add_value \
                              --task=d4rl-$dataset-v2 \
                              --transition_init_num=110 \
                              --transition_select_num=$num_model \
                              --dynamics_path=/model/$dataset-$num_model-seed-$seed.th \
                              --traj_num_to_infer=$num_traj_infer \
                              --belief_update_mode=$belief_mode \
                              --add_value_to_rt=$add_value \
                              --horizon=$horizon_step \
                              --q_lambda=$lambda \
                              --calibration=$cal \
                              --seed=$seed