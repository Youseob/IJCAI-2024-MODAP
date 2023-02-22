#!/bin/bash
# dataset=$1
init_num=$1
num_model=$2
# num_traj_infer=$4
# seed=$5
python examples/train_d4rl.py --algo_name=adv_bayrl \
                              --exp_name=$prior_reg-reg-adv_bay_rl-$num_model-ensemble-$belief_mode-$temp-add-value-$add_value \
                              --task=d4rl-$dataset-v2 \
                              --transition_init_num=$init_num \
                              --transition_select_num=$num_model \
                              --dynamics_path=/model/$dataset-$num_model-seed-$seed.th \
                              --traj_num_to_infer=$num_traj_infer \
                              --belief_update_mode=$belief_mode \
                              --temp=$temp \
                              --add_value_to_rt=$add_value \
                              --prior_reg_init=$prior_reg \
                              --seed=$seed