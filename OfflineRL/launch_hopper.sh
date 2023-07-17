#!/bin/bash
# dataset=hopper-medium
# init_num_model=7
# num_model=5
# seed=42
# H=10
# reward_type=mean_reward
# lam=0
# epoch_per_div_update=2
# weight=0.1
# real_data_ratio=0.1
# actor_lr=3e-5
# div_lr=3e-5
python examples/train_d4rl.py --algo_name=maple_div_v1 \
                              --exp_name=div-$num_model-model-$H-H-$weight-dw-$actor_lr-actor_lr-$div_lr-div_lr-$epoch_per_div_update-update-$seed \
                              --task=d4rl-$dataset-v2 \
                              --transition_init_num=$init_num_model \
                              --transition_select_num=$num_model \
                              --dynamics_save_path=/output/$dataset-$num_model-$seed-ckpt.th \
                              --horizon=$H \
                              --epoch_per_div_update=$epoch_per_div_update \
                              --real_data_ratio=$real_data_ratio \
                              --actor_lr=$actor_lr \
                              --div_lr=$div_lr \
                              --reward_type=$reward_type \
                              --lam=$lam \
                              --diversity_weight=$weight \
                              --save_path=/output/div-$num_model-model-$H-H-$weight-dw-$actor_lr-actor_lr-$div_lr-div_lr-$epoch_per_div_update-update-$seed-ckpt.th \
                              --seed=$seed