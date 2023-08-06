import argparse
import time
import subprocess
import concurrent.futures
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo_name", type=str, default="maple")
    parser.add_argument("--dataset", type=str, default="hopper-medium")
    #------
    parser.add_argument("--init_num_model", type=int, default=7)
    parser.add_argument("--num_model", type=int, default=5)
    parser.add_argument("--H", type=int, default=10)
    #-----
    parser.add_argument("--out_epochs", type=int, default=2400)
    #-----
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--real_data_ratio", type=float, default=0.05)
    parser.add_argument("--reward_type", type=str, default="penalized_reward")
    parser.add_argument("--lam", type=float, default=0)
    parser.add_argument("--num_worker", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    
    return parser.parse_args()

def main_parallel_run(args=get_args()):

    num_worker = args.num_worker
    start = time.time()
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_worker)
    
    runs = [f'python examples/train_d4rl.py \
            --algo_name={args.algo_name} \
            --exp_name={args.algo_name}-{args.num_model}-model-{args.H}-H-{seed} \
            --task=d4rl-{args.dataset}-v2 \
            --transition_init_num={args.init_num_model} \
            --transition_select_num={args.num_model} \
            --dynamics_path=/model/{args.dataset}-{args.num_model}-{seed}-ckpt.th \
            --horizon={args.H} \
            --out_epochs={args.out_epochs} \
            --actor_lr={args.actor_lr} \
            --reward_type={args.reward_type} \
            --lam={args.lam} \
            --save_path=/output/maple-{args.num_model}-model-{args.H}-H-{args.weight}-dw-{seed}-ckpt.th \
            --seed={seed}'
            for seed in range(args.seed, args.seed+num_worker)                  
            ]
    
    procs = []
    for i, single_run in enumerate(runs):
        procs.append(executor.submit(subprocess.run, single_run, shell=True, capture_output=True)) 

    for p in concurrent.futures.as_completed(procs):   
        print(p.result().args[-1] + '...completed')
        if p.result().returncode != 0:
            print('--Error at ' + p.result().args[-1])


    end = time.time()
    print("Done", end-start, 's')

if __name__ == '__main__':
    main_parallel_run()
