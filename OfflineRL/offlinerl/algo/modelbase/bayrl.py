import os
import torch
from torch.distributions import Normal, Dirichlet, Categorical
import numpy as np
from copy import deepcopy
from loguru import logger
# import ray

from offlinerl.utils.env import get_env
from offlinerl.algo.base import BaseAlgo
from collections import OrderedDict
from offlinerl.utils.data import Batch
from offlinerl.utils.net.common import MLP, Net
from offlinerl.utils.net.tanhpolicy import TanhGaussianPolicy
from offlinerl.utils.exp import setup_seed
import offlinerl.utils.loader as loader
from offlinerl.utils.net.terminal_check import is_terminal

from offlinerl.utils.data import ModelBuffer
from offlinerl.utils.net.model.ensemble import EnsembleTransition, GammaPriorNNEnsemble
from offlinerl.utils.net.maple_actor import Maple_actor
from offlinerl.utils.net.model.maple_critic import Maple_critic
from offlinerl.utils.simple_replay_pool import SimpleReplayTrajPool

import uuid
import wandb

def algo_init(args):
    logger.info('Run algo_init function')

    setup_seed(args['seed'])

    if args["obs_shape"] and args["action_shape"]:
        obs_shape, action_shape = args["obs_shape"], args["action_shape"]
    elif "task" in args.keys():
        from offlinerl.utils.env import get_env_shape, get_env_obs_act_spaces
        obs_shape, action_shape = get_env_shape(args['task'])
        obs_space, action_space = get_env_obs_act_spaces(args['task'])
        args["obs_shape"], args["action_shape"] = obs_shape, action_shape
        args['obs_space'], args['action_space'] = obs_space, action_space
    else:
        raise NotImplementedError

    args['data_name'] = args['task'][5:]

    if args["model_type"] == 'esnn':
        transition = EnsembleTransition(obs_shape, action_shape, args['hidden_layer_size'], args['transition_layers'],
                                        args['transition_init_num'], mode=args['mode']).to(args['device'])
    elif args["model_type"] =="ivgesnn":
        transition = GammaPriorNNEnsemble(obs_shape, action_shape, args['hidden_layer_size'], args['transition_layers'],
                                        args['transition_init_num'], mode=args['mode']).to(args['device'])
    
    transition_optim = torch.optim.AdamW(transition.parameters(), lr=args['transition_lr'], weight_decay=0.000075)


    # lstm_hidden_unit: dim(belief_vector)0
    ###################################################
    actor = Maple_actor(args['obs_shape'], args['action_shape'], lstm_hidden_unit=args["transition_select_num"]).to(args['device'])
    q1 = Maple_critic(args['obs_shape'], args['action_shape'], lstm_hidden_unit=args["transition_select_num"]).to(args['device'])
    q2 = Maple_critic(args['obs_shape'], args['action_shape'], lstm_hidden_unit=args["transition_select_num"]).to(args['device'])
    ###################################################

    actor_optim = torch.optim.Adam(actor.parameters(), lr=args['actor_lr'])
    critic_optim = torch.optim.Adam([*q1.parameters(), *q2.parameters()], lr=args['critic_lr'])

    log_alpha = torch.zeros(1, requires_grad=True, device=args['device'])
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=args["actor_lr"])

    return {
        "transition": {"net": transition, "opt": transition_optim},
        "actor": {"net": actor, "opt": actor_optim},
        "log_alpha": {"net": log_alpha, "opt": alpha_optimizer},
        "critic": {"net": [q1, q2], "opt": critic_optim},
    }


class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args
                
        wandb.init(
            config=self.args,
            project=self.args["task"], # "d4rl-halfcheetah-medium-v2"
            group=self.args["algo_name"], # "maple"
            name=self.args["exp_name"], 
            id=str(uuid.uuid4())
        )
        self.transition = algo_init['transition']['net']
        self.transition_optim = algo_init['transition']['opt']
        self.selected_transitions = None

        self.actor = algo_init['actor']['net']
        self.actor_optim = algo_init['actor']['opt']

        self.log_alpha = algo_init['log_alpha']['net']
        self.log_alpha_optim = algo_init['log_alpha']['opt']

        self.q1, self.q2 = algo_init['critic']['net']
        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)
        self.critic_optim = algo_init['critic']['opt']

        self.device = args['device']
        self.obs_space = args['obs_space']
        self.action_space = args['action_space']

        self.args['buffer_size'] = int(self.args['data_collection_per_epoch']) * self.args['horizon'] * 5
        self.args['target_entropy'] = - self.args['action_shape']
        self.args['model_pool_size'] = int(args['model_pool_size'])

    def train(self, train_buffer, val_buffer, callback_fn):
        self.transition.update_self(torch.cat((torch.Tensor(train_buffer["obs"]), torch.Tensor(train_buffer["obs_next"])), 0))
        if self.args['dynamics_path'] is not None:
            ckpt = torch.load(self.args['dynamics_path'], map_location='cpu')
            self.transition = ckpt["model"].to(self.device)
            self.transition_optim = ckpt["optim"]
            print("[ DEBUG ] load state dict model done")
        else:
            # self.transition.update_self(torch.cat((torch.Tensor(train_buffer["obs"]), torch.Tensor(train_buffer["obs_next"])), 0))
            self.train_transition(train_buffer)
            if self.args['dynamics_save_path'] is not None: 
                torch.save({'model': self.transition, 'optim': self.transition_optim}, self.args['dynamics_save_path'])
        if self.args['only_dynamics']:
            return
        self.transition.requires_grad_(False)
        ###################################################################################################################
        
        env_pool_size = int((train_buffer.shape[0]/self.args['horizon']) * 1.2)
        self.env_pool = SimpleReplayTrajPool(self.obs_space, self.action_space, self.args['horizon'],\
                                             self.args['transition_select_num'], env_pool_size)
        self.model_pool = SimpleReplayTrajPool(self.obs_space, self.action_space, self.args['horizon'],\
                                               self.args['transition_select_num'],self.args['model_pool_size'])

        loader.restore_pool_d4rl(self.env_pool, self.args['data_name'],adapt=True,\
                                 maxlen=self.args['horizon'], policy_hook=None,\
                                 value_hook=None, model_hook=self.transition,\
                                 belief_update_mode=self.args["belief_update_mode"], temp=self.args["temp"], \
                                 device=self.device, traj_num_to_infer=self.args["traj_num_to_infer"])
        torch.cuda.empty_cache()
        self.obs_max = train_buffer['obs'].max(axis=0)
        self.obs_min = train_buffer['obs'].min(axis=0)
        expert_range = self.obs_max - self.obs_min
        soft_expanding = expert_range * 0.05
        self.obs_max += soft_expanding
        self.obs_min -= soft_expanding
        self.obs_max = np.maximum(self.obs_max, 100)
        self.obs_min = np.minimum(self.obs_min, -100)

        self.rew_max = train_buffer['rew'].max()
        # self.args["lam"] == 0
        self.rew_min = train_buffer['rew'].min() - self.args['penalty_clip'] * self.args['lam']

        for i in range(self.args['out_train_epoch']):
            for _ in range(50000 // self.args["rollout_batch_size"]):
                uncertainty_mean, uncertainty_max = self.rollout_model(self.args['rollout_batch_size'])
            
            torch.cuda.empty_cache()

            train_loss = {}
            train_loss['policy_loss'] = 0
            train_loss['q_loss'] = 0
            train_loss['uncertainty_mean'] = uncertainty_mean
            train_loss['uncertainty_max'] = uncertainty_max
            for j in range(self.args['in_train_epoch']):
                batch = self.get_train_policy_batch(self.args['train_batch_size'])
                in_res = self.train_policy(batch)
                for key in in_res:
                    train_loss[key] = train_loss[key] + in_res[key]
            for k in train_loss:
                train_loss[k] = train_loss[k]/self.args['in_train_epoch']
            
            # evaluate in mujoco
            eval_loss = self.eval_policy()
            train_loss.update(eval_loss)
            torch.cuda.empty_cache()
            self.log_res(i, train_loss)

    def get_train_policy_batch(self, batch_size = None):
        batch_size = batch_size or self.args['train_batch_size']
        env_batch_size = int(batch_size * self.args['real_data_ratio'])
        model_batch_size = batch_size - env_batch_size

        env_batch = self.env_pool.random_batch(env_batch_size)

        if model_batch_size > 0:
            model_batch = self.model_pool.random_batch(model_batch_size)

            keys = set(env_batch.keys()) & set(model_batch.keys())
            batch = {k: np.concatenate((env_batch[k], model_batch[k]), axis=0) for k in keys}
        else:
            ## if real_ratio == 1.0, no model pool was ever allocated,
            ## so skip the model pool sampling
            batch = env_batch
        return batch

    def get_policy(self):
        return self.policy_gru , self.actor

    def get_meta_action(self, state, belief ,deterministic=False, out_mean_std=False):

        mu_res, action_res, log_p_res, std_res = self.actor(belief, state)
        # action_res = torch.squeeze(action_res, dim=1)
        # mu_res = torch.squeeze(mu_res, dim=1)
        # std_res = torch.squeeze(std_res, dim=1)

        if out_mean_std:
            return mu_res, action_res, std_res

        if deterministic:
            return mu_res
        else:
            return action_res

    def train_transition(self, buffer):
        data_size = len(buffer)
        val_size = min(int(data_size * 0.2) + 1, 1000)
        train_size = data_size - val_size
        train_splits, val_splits = torch.utils.data.random_split(range(data_size), (train_size, val_size))
        train_buffer = buffer[train_splits.indices]
        valdata = buffer[val_splits.indices]
        batch_size = self.args['transition_batch_size']

        if self.args["transition_epoch"]:
            for epoch in range(self.args["transition_epoch"]):
                idxs = np.random.randint(train_buffer.shape[0], size=[self.transition.ensemble_size, train_buffer.shape[0]])
                for batch_num in range(int(np.ceil(idxs.shape[-1] / batch_size))):
                    batch_idxs = idxs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
                    batch = train_buffer[batch_idxs]
                    self._train_transition(self.transition, batch, self.transition_optim)
                new_val_losses = list(self._eval_transition(self.transition, valdata, inc_var_loss=False).cpu().numpy())
                print(np.mean(new_val_losses))
                self.log_res(epoch, {"transition_val_loss" : np.mean(new_val_losses)})

        else:
            val_losses = np.array([float('inf') for i in range(self.transition.ensemble_size)])

            epoch = 0
            cnt = 0
            while True:
                epoch += 1
                idxs = np.random.randint(train_buffer.shape[0], size=[self.transition.ensemble_size, train_buffer.shape[0]])
                for batch_num in range(int(np.ceil(idxs.shape[-1] / batch_size))):
                    batch_idxs = idxs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
                    batch = train_buffer[batch_idxs]
                    self._train_transition(self.transition, batch, self.transition_optim)
                
                new_val_losses = self._eval_transition(self.transition, valdata, inc_var_loss=False).cpu().numpy()
                ###
                new_val_losses = np.floor(new_val_losses * 1000) / 1000                
                mask = new_val_losses < val_losses
                
                if mask.any():
                    new_val_losses < val_losses
                    indexes = np.where(mask)[0].tolist()
                    cnt = 0
                    print(new_val_losses[mask])
                    val_losses[mask] = new_val_losses[mask]
                    self.transition.update_save(indexes)

                else:
                    cnt += 1
                
                if epoch % 1000 == 0:
                    print(f"[ epoch {epoch}] print val_losses")
                    print(val_losses)
                
                if cnt >= 3:
                    print(f"[ epoch {epoch} ] Done training dynamics model")
                    print(val_losses)
                    break
                

            indexes = self._select_best_indexes(val_losses, n=self.args['transition_select_num'])
            self.transition.set_select(indexes)
        return self.transition

    def rollout_model(self,rollout_batch_size, deterministic=False):
        
        batch = self.env_pool.random_batch_for_initial(rollout_batch_size)
        obs = torch.from_numpy(batch['observations']).to(self.device)
        num_dynamics = len(self.transition.output_layer.select)

        belief = torch.from_numpy(batch["policy_hidden"]).to(self.device)
        
        obs_max = torch.tensor(self.obs_max).to(self.device)
        obs_min = torch.tensor(self.obs_min).to(self.device)
        rew_max = self.rew_max
        rew_min = self.rew_min

        uncertainty_list = []
        uncertainty_max = []

        current_nonterm = np.ones((len(obs)), dtype=bool)
        samples = None
        with torch.no_grad():
            model_indexes = None
            for i in range(self.args['horizon']):
                act = self.get_meta_action(obs, belief, deterministic)
                obs_action = torch.cat([obs,act], dim=-1) # (500000 : rollout_batch_size, 18)
                next_obs_dists = self.transition(obs_action)
                next_obses = next_obs_dists.sample() # (num_dynamics, rollout_batch_size, obs_dim)
                log_probs = torch.stack([Normal(next_obs_dists.mean[index], next_obs_dists.scale[index]).log_prob(next_obses).sum(-1) for index in range(num_dynamics)]) # num_dynamics
                # log_probs = next_obs_dists.log_prob(next_obses[None, ...].repeat(num_dynamics, 1, 1, 1)).sum(-1) # (num_dynamics, num_dynamics, rollout_batch_size)
                rewards = next_obses[:, :, -1:] # (num_dynamics, rollout_batch_size, obs_dim)
                next_obses = next_obses[:, :, :-1]

                next_obses_mode = next_obs_dists.mean[:, :, :-1]
                next_obs_mean = torch.mean(next_obses_mode, dim=0)
                diff = next_obses_mode - next_obs_mean
                disagreement_uncertainty = torch.max(torch.norm(diff, dim=-1, keepdim=True), dim=0)[0]
                aleatoric_uncertainty = torch.max(torch.norm(next_obs_dists.stddev, dim=-1, keepdim=True), dim=0)[0]
                uncertainty = disagreement_uncertainty if self.args['uncertainty_mode'] == 'disagreement' else aleatoric_uncertainty
                uncertainty = torch.clamp(uncertainty, max=self.args['penalty_clip'])
                uncertainty_list.append(uncertainty.mean().item())
                uncertainty_max.append(uncertainty.max().item())
                if model_indexes is None:
                    if self.args["uniform_rollout"]: 
                        model_indexes = np.random.randint(0, next_obses.shape[0], size=(obs.shape[0]))
                    # belief (rollout_batch_size, num_dynamics)
                    else: 
                        model_indexes = Categorical(belief).sample().cpu().numpy()
                next_obs = next_obses[model_indexes, np.arange(obs.shape[0])] # 50000, obs_dim
                reward = rewards[model_indexes, np.arange(obs.shape[0])]
                log_prob = torch.stack([log_probs[index][model_indexes, np.arange(obs.shape[0])] for index in range(num_dynamics)]) # num_dynamics, 500000

                term = is_terminal(obs.cpu().numpy(), act.cpu().numpy(), next_obs.cpu().numpy(), self.args['task'])
                next_obs = torch.clamp(next_obs, obs_min, obs_max)
                reward = torch.clamp(reward, rew_min, rew_max)
                log_prob = torch.clamp(log_prob, -20, 5.)
                next_belief = self.belief_update(belief, log_prob=log_prob)
                

                penalized_reward = reward - self.args['lam'] * uncertainty

                nonterm_mask = ~term.squeeze(-1)
                #nonterm_mask: 1-not done, 0-done
                samples = {'observations': obs.cpu().numpy(), 'actions': act.cpu().numpy(), 'next_observations': next_obs.cpu().numpy(),
                           'rewards': penalized_reward.cpu().numpy(), 'terminals': term,
                           'valid': current_nonterm.reshape(-1, 1),
                           'value_hidden': next_belief.cpu().numpy(), 'policy_hidden': belief.cpu().numpy()} 
                
                samples = {k: np.expand_dims(v, 1) for k, v in samples.items()}
                num_samples = samples['observations'].shape[0]
                index = np.arange(
                    self.model_pool._pointer, self.model_pool._pointer + num_samples) % self.model_pool._max_size
                for k in samples:
                    self.model_pool.fields[k][index, i] = samples[k][:, 0]
                current_nonterm = current_nonterm & nonterm_mask
                obs = next_obs
                belief = next_belief
            self.model_pool._pointer += num_samples
            self.model_pool._pointer %= self.model_pool._max_size
            self.model_pool._size = min(self.model_pool._max_size, self.model_pool._size + num_samples)
        return

    def train_policy(self, batch):
        batch['valid'] = batch['valid'].astype(int)
        lens = np.sum(batch['valid'], axis=1).squeeze(-1)
        max_len = np.max(lens)
        for k in batch:
            batch[k] = torch.from_numpy(batch[k][:,:max_len]).to(self.device)
        belief = batch['policy_hidden']
        next_belief = batch['value_hidden']

        q1 = self.q1(belief,batch['actions'],batch['observations'])
        q2 = self.q2(belief,batch['actions'],batch['observations'])
        valid_num = torch.sum(batch['valid'])
        
        with torch.no_grad():
            mu_target, act_target, log_p_act_target, std_target = self.actor(next_belief,\
                                                                             batch['next_observations'])
            q1_target = self.target_q1(next_belief,act_target,batch['next_observations'])
            q2_target = self.target_q2(next_belief,act_target,batch['next_observations'])
            Q_target = torch.min(q1_target,q2_target)
            alpha = torch.exp(self.log_alpha)
            Q_target = Q_target - alpha*torch.unsqueeze(log_p_act_target,dim=-1)
            Q_target = batch['rewards'] + self.args['discount']*(~batch['terminals'])*(Q_target)
            Q_target = torch.clip(Q_target,self.rew_min/(1-self.args['discount']),\
                                  self.rew_max/(1-self.args['discount']))

        q1_loss = torch.sum(((q1-Q_target)**2)*batch['valid'])/valid_num
        q2_loss = torch.sum(((q2-Q_target)**2)*batch['valid'])/valid_num
        q_loss = (q1_loss+q2_loss)/2

        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        self._sync_weight(self.target_q1, self.q1, soft_target_tau=self.args['soft_target_tau'])
        self._sync_weight(self.target_q2, self.q2, soft_target_tau=self.args['soft_target_tau'])

        mu_now, act_now, log_p_act_now, std_now = self.actor(belief, batch['observations'])
        log_p_act_now = torch.unsqueeze(log_p_act_now, dim=-1)

        if self.args['learnable_alpha']:
            # update alpha
            alpha_loss = - torch.sum(self.log_alpha * ((log_p_act_now+ \
                                                         self.args['target_entropy'])*batch['valid']).detach())/valid_num

            self.log_alpha_optim.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optim.step()
        
        q1_ = self.q1(belief, act_now, batch['observations'])
        q2_ = self.q2(belief, act_now, batch['observations'])
        min_q_ = torch.min(q1_, q2_)
        policy_loss = alpha*log_p_act_now - min_q_
        policy_loss = torch.sum(policy_loss*batch['valid'])/valid_num

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        res = {}
        res['policy_loss'] = policy_loss.cpu().detach().numpy()
        res['q_loss'] = q_loss.cpu().detach().numpy()
        return res

    def _select_best_indexes(self, metrics, n):
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        selected_indexes = [pairs[i][1] for i in range(n)]
        return selected_indexes

    def _train_transition(self, transition, data, optim):
        data.to_torch(device=self.device)
        ''' calculation in MOPO '''
        dist = transition(torch.cat([data['obs'], data['act']], dim=-1))
        loss = - dist.log_prob(torch.cat([data['obs_next'], data['rew']], dim=-1))
        loss = loss.sum()
        ''' calculation when not deterministic TODO: figure out the difference A: they are the same when Gaussian''' 
        loss += 0.01 * (2. * transition.max_logstd).sum() - 0.01 * (2. * transition.min_logstd).sum()

        optim.zero_grad()
        loss.backward()
        optim.step()

    def _eval_transition(self, transition, valdata, inc_var_loss=True):
        with torch.no_grad():
            valdata.to_torch(device=self.device)
            dist = transition(torch.cat([valdata['obs'], valdata['act']], dim=-1))
            if inc_var_loss:
                mse_losses = ((dist.mean - torch.cat([valdata['obs_next'], valdata['rew']], dim=-1)) ** 2 / (dist.variance + 1e-8)).mean(dim=(1, 2))
                logvar = 2 * transition.max_logstd - torch.nn.functional.softplus(2 * transition.max_logstd - torch.log(dist.variance))
                logvar = 2 * transition.min_logstd + torch.nn.functional.softplus(logvar - 2 * transition.min_logstd)
                var_losses = logvar.mean(dim=(1, 2))
                loss = mse_losses + var_losses
            else:
                loss = ((dist.mean - torch.cat([valdata['obs_next'], valdata['rew']], dim=-1)) ** 2).mean(dim=(1, 2))
            return loss

    def eval_policy(self):
        env = get_env(self.args['task'])
        eval_res = OrderedDict()
        res = self.test_on_real_env(self.args['number_runs_eval'], env)
        return res

    def test_on_real_env(self, number_runs, env):
        results = ([self.test_one_trail(env) for _ in range(number_runs)])
        rewards = [result[0] for result in results]
        episode_lengths = [result[1] for result in results]
        belief_10th = [result[2] for result in results]
        belief_49th = [result[3] for result in results]
        belief_99th = [result[4] for result in results]
        belief_last = [result[5] for result in results]

        rew_mean = np.mean(rewards)
        len_mean = np.mean(episode_lengths)
        res = OrderedDict()
        res["Reward_Mean_Env"] = rew_mean
        res["Eval_normalized_score"] = env.get_normalized_score(rew_mean)
        res["Length_Mean_Env"] = len_mean
        res["10th_Belief_Max"] = np.mean(belief_10th)
        res["49th_Belief_Max"] = np.mean(belief_49th)
        res["99th_Belief_Max"] = np.mean(belief_99th)
        res["Last_Belief_Max"] = np.mean(belief_last)

        return res

    def test_one_trail(self, env):
        env = deepcopy(env)
        with torch.no_grad():
            state, done = env.reset(), False
            belief = torch.ones((1,self.args['transition_select_num'])).to(self.device) / self.args['transition_select_num']
            # hidden_policy = torch.zeros((1,1,self.args['lstm_hidden_unit'])).to(self.device)
            rewards = 0
            lengths = 0
            state = state[np.newaxis]  
            state = torch.from_numpy(state).float().to(self.device)
            belief_10th = belief_50th = belief_100th = belief_last = 1. / self.args['transition_select_num']
            while not done:
                # hidden = (hidden_policy, lst_action)
                action = self.get_meta_action(state, belief, deterministic=True)
                use_action = action.cpu().numpy().reshape(-1)
                next_state, reward, done, _ = env.step(use_action)
                rewards += reward
                next_state = torch.from_numpy(next_state[None, ...]).float().to(self.device)
                reward = torch.from_numpy(reward[None, None, ...]).float().to(self.device)
                belief = self.belief_update(belief, state, action, next_state, reward)
                # lst_action = action
                state = next_state
                lengths += 1
                if lengths == 9:
                    belief_10th = belief.max().item()
                elif lengths == 49:
                    belief_50th = belief.max().item()
                elif lengths == 99:
                    belief_100th = belief.max().item()
                elif done:
                    belief_last = belief.max().item()
        return (rewards, lengths, belief_10th, belief_50th, belief_100th, belief_last)
    
    @torch.no_grad()
    def belief_update(self, belief, state=None, action=None ,next_state=None, reward=None, log_prob=None):
        # assert self.args["belief_update_mode"] in ['softmax', 'kl-reg', 'bay']
        if log_prob is None:
            obs_action = torch.cat([state, action], dim=-1) # bs, dim
            next_obs_dists = self.transition(obs_action) # bs, dim -> (num_dynamics, bs, dim)
            next_obses = torch.cat([next_state, reward], dim=-1) # bs, dim
            log_prob = next_obs_dists.log_prob(next_obses).sum(-1) # (num_dynamics, bs)
            log_prob = torch.clamp(log_prob, -20., 5.)
            
        if self.args["belief_update_mode"] == 'kl-reg':
            next_belief = belief * torch.exp(log_prob/self.args['temp']).T # bs, num_dynamics
            return 
        
        next_belief = belief * torch.exp(log_prob).T # bs, num_dynamics     
        if self.args["belief_update_mode"] == 'softmax':
            return torch.softmax(next_belief / self.args['temp'], dim=1)
        
        # 'bay"
        next_belief /= next_belief.sum(-1, keepdim=True)
        return next_belief
        