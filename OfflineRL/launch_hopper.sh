#!/bin/bash
# dataset=hopper-medium
# init_num_model=7
# num_model=5
# seed=42
# H=10
# epoch_per_div_update=5
# real_data_ratio=0.2
# weight=100
lam=0
reward_type=mean_reward
python examples/train_d4rl.py --algo_name=maple_div_v1 \
                              --exp_name=div-$num_model-model-$H-H-$weight-dw-$real_data_ratio-ratio-$epoch_per_div_update-update-$seed \
                              --task=d4rl-$dataset-v2 \
                              --transition_init_num=$init_num_model \
                              --transition_select_num=$num_model \
                              --dynamics_save_path=/output/$dataset-$num_model-$seed-ckpt.th \
                              --horizon=$H \
                              --epoch_per_div_update=$epoch_per_div_update \
                              --real_data_ratio=$real_data_ratio \
                              --reward_type=$reward_type \
                              --lam=$lam \
                              --diversity_weight=$weight \
                              --save_path=/output/div-$num_model-model-$H-H-$weight-dw-$seed-ckpt.th \
                              --seed=$seed