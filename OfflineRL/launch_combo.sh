#!/bin/bash
# task=Hopper-v3
# task_data_type="low"
# task_train_num=1000
# data_dir=None
# init_num_model=7
# num_model=5
# H=5
# beta=0.5
# out_epochs=1000
# seed=0
# num_worker=1
python required.py

python neorl_combo.py --algo_name=combo \
                      --task=$task \
                      --task_data_type=$task_data_type \
                      --task_train_num=100 \
                      --data_dir=$data_dir \
                      --init_num_model=$init_num_model \
                      --num_model=$num_model \
                      --H=$H \
                      --beta=$beta \
                      --out_epochs=$out_epochs \
                      --seed=$seed \
                      --num_worker=$num_worker

python neorl_combo.py --algo_name=combo \
                      --task=$task \
                      --task_data_type=$task_data_type \
                      --task_train_num=1000 \
                      --data_dir=$data_dir \
                      --init_num=$init_num_model \
                      --num_model=$num_model \
                      --H=$H \
                      --beta=$beta \
                      --out_epochs=$out_epochs \
                      --seed=$seed \
                      --num_worker=$num_worker