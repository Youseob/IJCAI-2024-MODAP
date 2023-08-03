#!/bin/bash
# task=Hopper-v3
# task_data_type="low"
# task_train_num=1000
# data_dir=None
# init_num_model=7
# num_model=5
# H=10
# epoch_per_div_update=1
# model_retrain_epochs=100 
# actor_lr=1e-4
# div_lr=3e-5
# real_data_ratio=0.05
# weight=100
# out_epochs=1000
lam=0
reward_type=mean_reward
# seed=0
# num_worker=4
python required.py

python neorl_worker.py --algo_name=maple_div_neorl \
                      --task=$task \
                      --task_data_type=$task_data_type \
                      --task_train_num=$task_train_num \
                      --data_dir=$data_dir \
                      --init_num=$init_num_model \
                      --num_model=$num_model \
                      --H=$H \
                      --out_epochs=$out_epochs \
                      --epoch_per_div_update=$epoch_per_div_update \
                      --model_retrain_epochs=$model_retrain_epochs \
                      --actor_lr=$actor_lr \
                      --div_lr=$div_lr \
                      --real_data_ratio=$real_data_ratio \
                      --weight=$weight \
                      --seed=$seed \
                      --num_worker=$num_worker