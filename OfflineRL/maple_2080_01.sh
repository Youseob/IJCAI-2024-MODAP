#!/bin/bash
# dataset=hopper-medium
# init_num_model=7
# num_model=5
# H=10
# epoch_per_div_update=5
# reward_type=penalized_reward
# lam=0.25
# transition_hidden_size
python required.py

python examples/train_d4rl.py --algo_name=maple \
                              --exp_name=maple-$num_model-model-$H-H-$reward_type-$lam-0 \
                              --task=d4rl-$dataset-v2 \
                              --transition_init_num=$init_num_model \
                              --transition_select_num=$num_model \
                              --transition_hidden_size=$transition_hidden_size \
                              --dynamics_save_path=/output/$dataset-$num_model-0-ckpt.th \
                              --save_path=/output/maple-$num_model-model-$H-H-0.ckpt.th  \
                              --horizon=$H \
                              --reward_type=$reward_type \
                              --lam=$lam \
                              --seed=0 &
python examples/train_d4rl.py --algo_name=maple \
                              --exp_name=maple-$num_model-model-$H-H-$reward_type-$lam-1 \
                              --task=d4rl-$dataset-v2 \
                              --transition_init_num=$init_num_model \
                              --transition_select_num=$num_model \
                              --transition_hidden_size=$transition_hidden_size \
                              --dynamics_save_path=/output/$dataset-$num_model-1-ckpt.th \
                              --save_path=/output/maple-$num_model-model-$H-H-1.ckpt.th  \
                              --horizon=$H \
                              --reward_type=$reward_type \
                              --lam=$lam \
                              --seed=1