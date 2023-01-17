#!/bin/bash
for dataset in $seq walker2d-random; do
    for seed in $seed 0; do
    for epsilon in $seq 0.9; do
            for beta in $seq 100; do
            CUDA_VISIBLE_DEVICES=1 python examples/train_d4rl.py --algo_name=bayrl_cvar --exp_name=bayrl_cvar_10-$epsilon-beta-$beta --task=d4rl-$dataset-v2 --dynamics_path=$dataset-20-seed-$seed.th --out_train_epoch=300 --worst_percentil=$epsilon --soft_belief_temp=$beta --seed=$seed
            done
        done
    done
done