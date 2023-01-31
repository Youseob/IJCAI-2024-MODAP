import wandb
import gym
import d4rl

for env in ['halfcheetah']:
    for dset in ['medium', 'medium-expert']:
        dset_name = env+'_'+dset.replace('-', '_')+'-v2'
        env_name = dset_name.replace('_', '-')
        env = gym.make(env_name)
        dataset = env.get_dataset()