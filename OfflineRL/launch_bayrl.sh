#!/bin/bash
init_num=110
num_model=100
python examples/train_d4rl.py --algo_name=bayrl_v2 \
                              --exp_name=bay_rl \
                              --task=d4rl-$dataset-v2 \
                              --transition_init_num=$init_num \
                              --transition_select_num=$num_model \
                              --dynamics_path=/model/$dataset-$num_model-seed-$seed.th \
                              --traj_num_to_infer=$num_traj_infer \
                              --seed=$seed