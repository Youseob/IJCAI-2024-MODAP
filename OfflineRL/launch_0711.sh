#!/bin/bash
# dataset=halfcheetah-medium
# init_num_model=7
# num_model=5
# seed=42
# H=5
# weight=0.1
# reward_type=penalized_reward
# lam=0.25
# epoch_per_div_update=3
python examples/train_d4rl.py --algo_name=maple_div_v1 \
                              --exp_name=div-$num_model-model-$H-H-$weight-dw-$reward_type-$lam \
                              --task=d4rl-$dataset-v2 \
                              --transition_init_num=$init_num_model \
                              --transition_select_num=$num_model \
                              --dynamics_save_path=/output/$dataset-$num_model-$seed-ckpt.th \
                              --horizon=$H \
                              --reward_type=$reward_type \
                              --lam=$lam \
                              --diversity_weight=$weight \
                              --epoch_per_div_update=$epoch_per_div_update \
                              --save_path=/output/div-$num_model-model-$H-H-$weight-dw-$seed.ckpt.th \
                              --seed=$seed