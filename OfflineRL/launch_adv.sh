#!/bin/bash
python examples/train_d4rl.py --algo_name=adv_bayrl_v2 \
                              --exp_name=adv-$belief_mode-$temp-add-value-$add_value-b-lr-$b_lr-r_scale-$r_scale \
                              --task=d4rl-$dataset-v2 \
                              --transition_init_num=110 \
                              --transition_select_num=$num_model \
                              --dynamics_path=/model/$dataset-$num_model-seed-$seed.th \
                              --traj_num_to_infer=$num_traj_infer \
                              --belief_update_mode=$belief_mode \
                              --temp=$temp \
                              --add_value_to_rt=$add_value \
                              --b_lr=$b_lr \
                              --reward_scale=$r_scale \
                              --seed=$seed