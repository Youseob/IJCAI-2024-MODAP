#!/bin/bash
# dataset=halfcheetah-medium
# num_model=5
# seed=42
# H=5
# weight=0.1
python examples/train_d4rl.py --algo_name=maple_div \
                              --exp_name=maple-$num_model-model-$H-H-$weight-dw-$seed \
                              --task=d4rl-$dataset-v2 \
                              --transition_init_num=$num_model \
                              --transition_select_num=$num_model \
                              --dynamics_save_path=/output/$dataset-$num_model-$seed-ckpt.th \
                              --horizon=$H \
                              --diversity_weight=$weight \
                              --save_path=/output/maple-$num_model-model-$H-H-$weight-dw-$seed.ckpt.th \
                              --seed=$seed