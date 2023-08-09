import argparse
import time
import subprocess
import concurrent.futures
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo_name", type=str, default="combo")
    parser.add_argument("--task", type=str, default="Hopper-v3")
    parser.add_argument("--task_data_type", type=str, default="low")
    parser.add_argument("--task_train_num", type=int, default=100)
    parser.add_argument("--data_dir", type=str, default="/input")
    #------
    parser.add_argument("--init_num_model", type=int, default=7)
    parser.add_argument("--num_model", type=int, default=5)
    parser.add_argument("--H", type=int, default=5)
    #-----
    parser.add_argument("--out_epochs", type=int, default=500)
    #-----
    parser.add_argument("--beta", type=float, default=0)
    #-----
    parser.add_argument("--num_worker", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    
    return parser.parse_args()

def main_parallel_run(args=get_args()):
    num_worker = args.num_worker
    start = time.time()
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_worker)
    
    runs = [f'python examples/train_task.py \
            --algo_name={args.algo_name} \
            --exp_name={args.num_model}-model-{args.H}-H-{args.beta}-beta-{seed} \
            --task={args.task} \
            --data_dir={args.data_dir} \
            --task_data_type={args.task_data_type} \
            --task_train_num={args.task_train_num} \
            --transition_init_num={args.init_num_model} \
            --transition_select_num={args.num_model} \
            --dynamics_path=/model/{args.task}-{args.task_data_type}-{args.task_train_num}-model-{args.num_model}-{seed}-ckpt.th \
            --horizon={args.H} \
            --base_beta={args.beta} \
            --seed={seed}'
            for seed in range(args.seed, args.seed+num_worker)                  
            ]
    
    procs = []
    for i, single_run in enumerate(runs):
        procs.append(executor.submit(subprocess.run, single_run, shell=True, capture_output=True)) # vessl True 

    for p in concurrent.futures.as_completed(procs):   
        print(p.result().args[-1] + '...completed')
        if p.result().returncode != 0:
            print('--Error at ' + p.result().args[-1])


    end = time.time()
    print("Done", end-start, 's')

if __name__ == '__main__':
    main_parallel_run()
