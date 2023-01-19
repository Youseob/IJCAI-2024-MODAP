#!/bin/bash
for dataset in $seq walker2d-random; do
    for seed in $seq 0; do
        python examples/train_d4rl.py --algo_name=bayrl_cvar --exp_name=train --train_dynamics=True --task=d4rl-$dataset-v2 --dynamics_save_path=$dataset-20-seed-$seed.th --seed=$seed
    for epsilon in $seq 0.9 0.7 0.5; do
            for beta in $seq 100 1 10; do
                python examples/train_d4rl.py --algo_name=bayrl_cvar --exp_name=bayrl_cvar_10-$epsilon-beta-$beta --task=d4rl-$dataset-v2 --dynamics_path=$dataset-20-seed-$seed.th --out_train_epoch=300 --worst_percentil=$epsilon --soft_belief_temp=$beta --seed=$seed
            done
        done
    done
done