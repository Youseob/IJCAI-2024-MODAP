#!/bin/bash
# dataset=halfcheetah-medium
# num_model=5
# seed=42
# H=5
python examples/train_d4rl.py --algo_name=maple \
                              --exp_name=maple-$num_model-mode-$seed \
                              --task=d4rl-$dataset-v2 \
                              --transition_init_num=$num_model \
                              --transition_select_num=$num_model \
                              --save_path=/output/$dataset-$num_model-$seed-ckpt.th \
                              --horizon=$H \
                              --seed=$seed