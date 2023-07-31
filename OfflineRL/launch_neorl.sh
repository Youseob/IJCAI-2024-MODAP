#!/bin/bash
# dataset=Hopper-v3
init_num_model=7
num_model=5
seed=0
H=10
reward_type=penalized_reward
lam=0.25
transition_hidden_size=400
WANDB_MODE=disabled python examples/train_task.py --algo_name=maple_div_neorl \
                              --exp_name=maple-$num_model-model-$H-H-$reward_type-$lam \
                              # --dynamics_path=/output/$dataset-$num_model-$seed-ckpt.th \
                            #   --task=d4rl-$dataset-v2 \
                            #   --transition_init_num=$init_num_model \
                            #   --transition_select_num=$num_model \
                            #   --transition_hidden_size=$transition_hidden_size \
                            #   --save_path=/output/maple-$num_model-model-$H-H-$weight-dw-$seed.ckpt.th  \
                            #   --horizon=$H \
                            #   --reward_type=$reward_type \
                            #   --lam=$lam \
                            #   --seed=0