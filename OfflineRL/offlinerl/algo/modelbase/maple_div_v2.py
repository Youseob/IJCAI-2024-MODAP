import torch
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
from offlinerl.utils.net.model.ensemble import EnsembleTransition
from offlinerl.utils.net.model_GRU import GRU_Model
from offlinerl.utils.net.maple_actor import Maple_actor
from offlinerl.utils.net.model.maple_critic import Maple_critic
from offlinerl.utils.simple_replay_pool import SimpleReplayTrajPool
from offlinerl.utils import loader
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

    transition = EnsembleTransition(obs_shape, action_shape, args['transition_hidden_size'], args['transition_layers'],
                                    args['transition_init_num'], mode=args['mode']).to(args['device'])
    transition_optim = torch.optim.AdamW(transition.parameters(), lr=args['transition_lr'], weight_decay=0.000075)
    div_transition_optim = torch.optim.AdamW(transition.parameters(), lr=args['div_lr'], weight_decay=0.000075)

    policy_gru = GRU_Model(args['obs_shape'], args['action_shape'], args['device'],args['lstm_hidden_unit']).to(args['device'])
    value_gru = GRU_Model(args['obs_shape'], args['action_shape'], args['device'], args['lstm_hidden_unit']).to(args['device'])
    actor = Maple_actor(args['obs_shape'], args['action_shape']).to(args['device'])
    q1 = Maple_critic(args['obs_shape'], args['action_shape']).to(args['device'])
    q2 = Maple_critic(args['obs_shape'], args['action_shape']).to(args['device'])

    actor_optim = torch.optim.Adam([*policy_gru.parameters(),*actor.parameters()], lr=args['actor_lr'])

    log_alpha = torch.zeros(1, requires_grad=True, device=args['device'])
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=args["actor_lr"])

    critic_optim = torch.optim.Adam([*value_gru.parameters(),*q1.parameters(), *q2.parameters()], lr=args['critic_lr'])

    return {
        "transition": {"net": transition, "opt": (transition_optim, div_transition_optim)},
        "actor": {"net": [policy_gru,actor], "opt": actor_optim},
        "log_alpha": {"net": log_alpha, "opt": alpha_optimizer},
        "critic": {"net": [value_gru, q1, q2], "opt": critic_optim},
    }


class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args        
        
        wandb.init(
            config=self.args,
            project="20230724-"+self.args["task"], # "d4rl-halfcheetah-medium-v2"
            group=self.args["algo_name"], # "maple"
            name=self.args["exp_name"], 
            id=str(uuid.uuid4())
        )
        self.transition = algo_init['transition']['net']
        self.transition_optim, self.div_transition_optim = algo_init['transition']['opt']
        self.selected_transitions = None

        self.policy_gru, self.actor = algo_init['actor']['net']
        self.actor_optim = algo_init['actor']['opt']

        self.log_alpha = algo_init['log_alpha']['net']
        self.log_alpha_optim = algo_init['log_alpha']['opt']

        self.value_gru, self.q1, self.q2 = algo_init['critic']['net']
        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)
        self.critic_optim = algo_init['critic']['opt']

        self.device = args['device']
        self.obs_space = args['obs_space']
        self.action_space = args['action_space']

        # self.args['buffer_size'] = int(self.args['data_collection_per_epoch']) * self.args['horizon'] * 5
        self.args['target_entropy'] = - self.args['action_shape']
        self.args['model_pool_size'] = int(args['model_pool_size'])

        print(f"[ DEBUG ] exp_name: {self.args['exp_name']}")

    def train(self, train_buffer, val_buffer, callback_fn):
        
        self.transition.update_self(torch.cat((torch.Tensor(train_buffer["obs"]), torch.Tensor(train_buffer["obs_next"])), 0))
        if self.args['dynamics_path'] is not None:
            ckpt = torch.load(self.args['dynamics_path'], map_location='cpu')
            self.transition = ckpt["model"].to(self.device)
            print("[ DEBUG ] load state dict model done")
            # self.transition = torch.load(self.args['dynamics_path'], map_location='cpu').to(self.device)
        else:
            self.train_transition(train_buffer)
            if self.args['dynamics_save_path'] is not None: 
                torch.save({'model': self.transition, 'optim': self.transition_optim}, self.args['dynamics_save_path'])
        
        if self.args['only_dynamics']:
            return
        
        env_pool_size = int((train_buffer.shape[0] / self.args['horizon']) * 1.2)
        self.env_pool = SimpleReplayTrajPool(self.obs_space, self.action_space, self.args['horizon'],\
                                             self.args['lstm_hidden_unit'], env_pool_size)
        self.model_pool = SimpleReplayTrajPool(self.obs_space, self.action_space, self.args['horizon'],\
                                               self.args['lstm_hidden_unit'], self.args['model_pool_size'])

        loader.restore_pool_d4rl(self.env_pool, self.args['data_name'],adapt=True,\
                                 maxlen=self.args['horizon'],policy_hook=self.policy_gru,\
                                 value_hook=self.value_gru, device=self.device)
        torch.cuda.empty_cache()
        
        self.obs_max = train_buffer['obs'].max(axis=0)
        self.obs_min = train_buffer['obs'].min(axis=0)
        expert_range = self.obs_max - self.obs_min
        soft_expanding = expert_range * 0.05
        self.obs_max += soft_expanding
        self.obs_min -= soft_expanding
        self.obs_max = torch.from_numpy(np.maximum(self.obs_max, 100)).float().to(self.device)
        self.obs_min = torch.from_numpy(np.minimum(self.obs_min, -100)).float().to(self.device)

        self.rew_max = train_buffer['rew'].max()
        self.rew_min = train_buffer['rew'].min() - self.args['penalty_clip'] * self.args['lam']
        # self.rew_min = train_buffer['rew'].min() 

        # log
        policy_log, model_log = {}, {}
        epoch = 0
        out_epochs = int(1000 / self.args["epoch_per_div_update"])
        for out_epoch in range(out_epochs):
            # train policy
            for epoch in range(epoch + 1, epoch + self.args["epoch_per_div_update"] + 1):
                self.rollout_model(self.args['rollout_batch_size'])
                torch.cuda.empty_cache()
                
                policy_log["Policy_Train/policy_loss"] = 0
                policy_log["Policy_Train/q_loss"] = 0
                policy_log["Policy_Train/q_val"] = 0
                for _ in range(self.args['policy_train_epochs']):
                    batch = self._get_train_policy_batch(self.args['train_batch_size'])
                    policy_res = self.train_policy(batch)
                    for key in policy_res:
                        policy_log[key] = policy_log[key] + policy_res[key]
               
                # average policy_res 
                for k in policy_res:
                    policy_log[k] /= self.args['policy_train_epochs']
                
                # evaluate in mujoco
                eval_res = self.eval_policy(self.args["number_runs_eval"])
                policy_log.update(eval_res)
                self.log_res(epoch, policy_log)
            
                # if epoch % 4 == 0:
                if epoch % self.args["epoch_per_div_update"] == 0:
                    loader.reset_hidden_state(self.env_pool, self.args['data_name'],\
                                    maxlen=self.args['horizon'], policy_hook=self.policy_gru,\
                                    value_hook=self.value_gru, device=self.device)
                    torch.cuda.empty_cache()
            
            # train dynamics
            if (out_epoch + 1) < self.args['out_epochs']:
                model_log["Model_Train/mle_loss"] = 0
                model_log["Model_Train/div_loss"] = 0
                for _ in range(self.args["model_retrain_epochs"]):
                    model_res = self.retrain_transition()   
                    for key in model_res:
                        model_log[key] += model_res[key]
                for key in model_res:
                    model_log[key] /= self.args["model_retrain_epochs"]
                self.log_res(epoch, model_log)
        
        if self.args["save_path"] is not None:
            torch.save({'actor': self.actor, 'q1': self.q1, 'q2': self.q2, 'model': self.transition}, self.args['save_path'])
        
    def _get_train_policy_batch(self, batch_size=None):
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

    def get_meta_action(self, state, hidden, deterministic=False, out_mean_std=False):
        if len(state.shape) == 2:
            state = torch.unsqueeze(state, dim=1)
        lens = [1] *state.shape[0]
        hidden_policy, lst_action = hidden
        if len(hidden_policy.shape) == 2:
            hidden_policy = torch.unsqueeze(hidden_policy, dim=0)
        if len(lst_action.shape) == 2:
            lst_action = torch.unsqueeze(lst_action, dim=1)
        
        hidden_policy_res = self.policy_gru(state, lst_action, hidden_policy, lens)
        mu_res, action_res, log_p_res, std_res = self.actor(hidden_policy_res, state)
        hidden_policy_res = torch.squeeze(hidden_policy_res, dim=1)
        action_res = torch.squeeze(action_res, dim=1)
        mu_res = torch.squeeze(mu_res, dim=1)
        std_res = torch.squeeze(std_res, dim=1)

        if out_mean_std:
            return mu_res, action_res, std_res, hidden_policy_res

        if deterministic:
            return mu_res, hidden_policy_res
        else:
            return action_res , hidden_policy_res

    def train_transition(self, buffer):
        data_size = len(buffer)
        val_size = min(int(data_size * 0.2) + 1, 1000)
        train_size = data_size - val_size
        train_splits, val_splits = torch.utils.data.random_split(range(data_size), (train_size, val_size))
        train_buffer = buffer[train_splits.indices]
        valdata = buffer[val_splits.indices]
        batch_size = self.args['transition_batch_size']

        val_losses = [float('inf') for i in range(self.transition.ensemble_size)]

        epoch = 0
        cnt = 0
        while True:
            epoch += 1
            idxs = np.random.randint(train_buffer.shape[0], size=[self.transition.ensemble_size, train_buffer.shape[0]])
            for batch_num in range(int(np.ceil(idxs.shape[-1] / batch_size))):
                batch_idxs = idxs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
                batch = train_buffer[batch_idxs]
                self._train_transition(self.transition, batch, self.transition_optim)
            new_val_losses = list(self._eval_transition(self.transition, valdata, inc_var_loss=False).cpu().numpy())
            print(new_val_losses)

            indexes = []
            for i, new_loss, old_loss in zip(range(len(val_losses)), new_val_losses, val_losses):
                if new_loss < old_loss:
                    indexes.append(i)
                    val_losses[i] = new_loss

            if len(indexes) > 0:
                self.transition.update_save(indexes)
                cnt = 0

            else:
                cnt += 1

            if cnt >= 5:
                break

        indexes = self._select_best_indexes(val_losses, n=self.args['transition_select_num'])
        self.transition.set_select(indexes)
        return self.transition

    @torch.no_grad()
    def rollout_model(self,rollout_batch_size, deterministic=False):
        batch = self.env_pool.random_batch_for_initial(rollout_batch_size)
        num_dynamics = self.args["transition_select_num"]
        hidden_value_init = batch['value_hidden'] # rollout_batch_size, dim
        hidden_policy_init = batch['policy_hidden']
        sum_reward = np.zeros((rollout_batch_size, self.args['N'], 1))

        _obs = torch.zeros(rollout_batch_size, self.args['N'], self.args['horizon'], self.obs_space.shape[0]).float().to(self.device)
        _next_obs = torch.zeros(rollout_batch_size, self.args['N'], self.args['horizon'], self.obs_space.shape[0]).float().to(self.device)
        _act = torch.zeros(rollout_batch_size, self.args['N'], self.args["horizon"], self.action_space.shape[0]).float().to(self.device)
        _reward = torch.zeros(rollout_batch_size, self.args['N'], self.args["horizon"], 1).float().to(self.device)
        _valid = np.zeros((rollout_batch_size, self.args['N'], self.args["horizon"], 1))
        _term = np.zeros((rollout_batch_size, self.args['N'], self.args["horizon"], 1))
        
        for n in range(self.args['N']):
            model_indexes = np.random.randint(0, num_dynamics, size=(rollout_batch_size))
            obs = torch.from_numpy(batch['observations']).to(self.device)
            lst_action = torch.from_numpy(batch['last_actions']).to(self.device)
            hidden_policy = torch.from_numpy(hidden_policy_init).to(self.device)
            hidden = (hidden_policy, lst_action)
            current_nonterm = np.ones((len(obs)), dtype=bool)
            for i in range(self.args['horizon']):
                act, hidden_policy = self.get_meta_action(obs, hidden, deterministic)
                obs_action = torch.cat([obs,act], dim=-1) # (500000 : rollout_batch_size, 18)
                next_obs_dists = self.transition(obs_action)
                next_obses = next_obs_dists.sample()[:, :, :-1] # (num_dynamics, rollout_batch_size, obs_dim)

                rewards, rewards_mean, rewards_scale = self.transition.saved_forward(obs_action, only_reward=True) # num_dynamics, rollout_batch, 1
                if self.args["reward_type"] == "mean_reward":
                    reward = rewards_mean.mean(0) # rollbout_batch , 1
                elif self.args["reward_type"] == "sample_reward":
                    reward = rewards[model_indexes, np.arange(rollout_batch_size)] # rollout_batch, 1


                next_obs = next_obses[model_indexes, np.arange(rollout_batch_size)]
                term = is_terminal(obs.cpu().numpy(), act.cpu().numpy(), next_obs.cpu().numpy(), self.args['task'])
                next_obs = torch.clamp(next_obs, self.obs_min, self.obs_max)
                reward = torch.clamp(reward, self.rew_min, self.rew_max)
                sum_reward[:, n] += (self.args["discount"]**i) * reward.cpu().numpy() * current_nonterm.reshape(-1, 1)
                nonterm_mask = ~term.squeeze(-1)
                
                _obs[:, n, i] = obs
                _next_obs[:, n, i] = next_obs
                _act[:, n, i] = act                
                _reward[:, n, i] = reward
                _term[:, n , i] = term
                _valid[:, n, i] = current_nonterm.reshape(-1, 1)
                _term[:, n, i] = term
            
                current_nonterm = current_nonterm & nonterm_mask
                obs = next_obs
                lst_action = act
                hidden = (hidden_policy, lst_action)
            
        worst_num_traj = int(self.args['N'] * self.args["worst_percentil"])
        worst_x_ind = np.arange(rollout_batch_size).repeat(worst_num_traj)
        worst_y_in = np.argsort(sum_reward, axis=1)[:, :worst_num_traj, :].reshape(-1) # rollout_batch_size, N*portion, 1 --> rollout_batch_size * N * portion, 
        
        num_samples = worst_num_traj * rollout_batch_size
        index = np.arange(self.model_pool._pointer, self.model_pool._pointer + worst_num_traj * rollout_batch_size) % self.model_pool._max_size
        self.model_pool.fields["observations"][index] = _obs[worst_x_ind, worst_y_in].cpu().numpy() # roll_bs * wort_num_traj, horizon, dim
        self.model_pool.fields["actions"][index] = _act[worst_x_ind, worst_y_in].cpu().numpy() # roll_bs * wort_num_traj, horizon, dim
        self.model_pool.fields["next_observations"][index] = _next_obs[worst_x_ind, worst_y_in].cpu().numpy() # roll_bs * wort_num_traj, horizon, dim
        self.model_pool.fields["rewards"][index] = _reward[worst_x_ind, worst_y_in].cpu().numpy() # roll_bs * wort_num_traj, horizon, dim
        self.model_pool.fields["terminals"][index] = _term[worst_x_ind, worst_y_in] # roll_bs *  wort_num_traj, horizon, dim
        self.model_pool.fields["valid"][index] = _valid[worst_x_ind, worst_y_in] # roll_bs * wort_num_traj, horizon, dim
        self.model_pool.fields["value_hidden"][index] =  np.tile(hidden_value_init[:, None, :], (worst_num_traj, self.args['horizon'], 1)) # roll_bs * wort_num_traj, horizon, dim
        self.model_pool.fields["policy_hidden"][index] = np.tile(hidden_policy_init[:, None, :], (worst_num_traj, self.args['horizon'], 1)) # roll_bs * wort_num_traj, horizon, dim
        
        self.model_pool._pointer += num_samples
        self.model_pool._pointer %= self.model_pool._max_size
        self.model_pool._size = min(self.model_pool._max_size, self.model_pool._size + num_samples)
        return
     
    def train_policy(self, batch):
        ################################################################
        batch['valid'] = batch['valid'].astype(int)
        lens = np.sum(batch['valid'], axis=1).squeeze(-1) # (batch_size)
        max_len = np.max(lens)
        for k in batch:
            batch[k] = torch.from_numpy(batch[k][: ,:max_len]).to(self.device)
        value_hidden = batch['value_hidden'][ :, 0]
        policy_hidden = batch['policy_hidden'][ :, 0]
        value_state = self.value_gru(batch['observations'], batch['last_actions'], value_hidden, lens) # (batch_size, H, dim)
        policy_state = self.policy_gru(batch['observations'], batch['last_actions'], policy_hidden, lens) # (batch_size, H, dim)
        lens_next = torch.ones(len(lens)).int() # batch_size
        next_value_hidden = value_state[:,-1] # batch_size, dim
        next_policy_hidden = policy_state[:,-1] # batch_size, dim
        value_state_next = self.value_gru(batch['next_observations'][:,-1:], \
                                          batch['last_actions'][:,-1:],next_value_hidden,lens_next) # (batch_size, 1, dim)
        policy_state_next = self.policy_gru(batch['next_observations'][:,-1:],\
                                            batch['last_actions'][:,-1:], next_policy_hidden, lens_next) # (batch_size, 1, dim)

        value_state_next = torch.cat([value_state[:,1:],value_state_next],dim=1) # (batch_size, H, dim)
        policy_state_next = torch.cat([policy_state[:,1:],policy_state_next],dim=1) # (batch_size, H, dim)

        q1 = self.q1(value_state, batch['actions'], batch['observations'])
        q2 = self.q2(value_state, batch['actions'], batch['observations'])
        valid_num = torch.sum(batch['valid'])
        
        with torch.no_grad():
            mu_target, act_target, log_p_act_target, std_target = self.actor(policy_state_next,\
                                                                             batch['next_observations'])
            q1_target = self.target_q1(value_state_next, act_target, batch['next_observations'])
            q2_target = self.target_q2(value_state_next, act_target, batch['next_observations'])
            Q_target = torch.min(q1_target, q2_target)
            alpha = torch.exp(self.log_alpha)
            Q_target = Q_target - alpha * torch.unsqueeze(log_p_act_target, dim=-1)
            Q_target = batch['rewards'] + self.args['discount'] * (~batch['terminals']) * (Q_target)
            Q_target = torch.clip(Q_target, self.rew_min / (1-self.args['discount']),\
                                  self.rew_max / (1 - self.args['discount']))

        q1_loss = torch.sum(((q1 - Q_target) ** 2) * batch['valid']) / valid_num
        q2_loss = torch.sum(((q2 - Q_target) ** 2) * batch['valid']) / valid_num
        q_loss = (q1_loss + q2_loss) / 2

        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        self._sync_weight(self.target_q1, self.q1, soft_target_tau=self.args['soft_target_tau'])
        self._sync_weight(self.target_q2, self.q2, soft_target_tau=self.args['soft_target_tau'])

        mu_now, act_now, log_p_act_now, std_now = self.actor(policy_state, batch['observations'])
        log_p_act_now = torch.unsqueeze(log_p_act_now, dim=-1)

        if self.args['learnable_alpha']:
            # update alpha
            alpha_loss = -torch.sum(self.log_alpha * ((log_p_act_now + \
                                                         self.args['target_entropy']) * batch['valid']).detach())/valid_num
            self.log_alpha_optim.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optim.step()
        
        q1_ = self.q1(value_state.detach(), act_now, batch['observations'])
        q2_ = self.q2(value_state.detach(), act_now, batch['observations'])
        min_q_ = torch.min(q1_, q2_)
        policy_loss = alpha * log_p_act_now - min_q_
        policy_loss = torch.sum(policy_loss * batch['valid']) / valid_num

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        res = {'Policy_Train/policy_loss': policy_loss.cpu().item(),
               'Policy_Train/q_loss': q_loss.cpu().item(),
               'Policy_Train/q_val': torch.mean(min_q_).cpu().item()}
        return res

    def retrain_transition(self):
        div_loss = self._dynamics_diversity_loss(self.args["div_rollout_batch_size"], deterministic=False)
        mle_loss = self._dynamics_mle_loss(self.args["mle_batch_size"])
        loss = self.args["diversity_weight"] * div_loss +  mle_loss
        
        self.div_transition_optim.zero_grad()
        loss.backward()
        self.div_transition_optim.step()
        
        return {
            "Model_Train/mle_loss": mle_loss.cpu().item(),
            "Model_Train/div_loss": div_loss.cpu().item()
            }
    
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

    def _dynamics_mle_loss(self, batch_size):
        batch = self.env_pool.random_batch_for_initial(batch_size)
        obs = torch.from_numpy(batch['observations']).to(self.device)
        act = torch.from_numpy(batch['actions']).to(self.device)
        obs_next = torch.from_numpy(batch["next_observations"]).to(self.device)
        reward = torch.from_numpy(batch['rewards']).to(self.device)
        dist = self.transition(torch.cat([obs, act], dim=-1))
        loss = - dist.log_prob(torch.cat([obs_next, reward], dim=-1))
        loss = loss.sum()
        ''' calculation when not deterministic TODO: figure out the difference A: they are the same when Gaussian''' 
        loss += 0.01 * (2. * self.transition.max_logstd).sum() - 0.01 * (2. * self.transition.min_logstd).sum()
        return loss
    
    def _dynamics_diversity_loss(self, rollout_batch_size, deterministic=True):
        batch = self.env_pool.random_batch_for_initial(rollout_batch_size)
        num_dynamics = self.args["transition_select_num"]
        
        obs = torch.from_numpy(batch['observations']).to(self.device)
        lst_action = torch.from_numpy(batch['last_actions']).to(self.device)
        hidden_policy = torch.from_numpy(batch['policy_hidden']).to(self.device)
        hidden = (hidden_policy, lst_action)

        obs_max = torch.tensor(self.obs_max).to(self.device)
        obs_min = torch.tensor(self.obs_min).to(self.device)
        
        model_indexes = np.random.randint(0, num_dynamics, size=(rollout_batch_size))
        traj_log_probs = torch.zeros(num_dynamics, rollout_batch_size).float().to(self.device)

        for h in range(self.args['horizon']):
            with torch.no_grad():
                act, hidden_policy = self.get_meta_action(obs, hidden, deterministic)
            obs_action = torch.cat([obs, act], dim=-1) # (rollout_batch_size, 18)
            next_obs_dists = self.transition(obs_action)
            next_obses = next_obs_dists.sample() # (num_dynamics, rollout_batch_size, obs_dim + 1)
            
            next_obs = next_obses[model_indexes, np.arange(rollout_batch_size)] # rollout_batch_size, obs_dim + 1
            traj_log_probs += next_obs_dists.log_prob(next_obs)[:, :, :-1].sum(-1) # num_dynamics, rollout_batch_size
            next_obs = torch.clamp(next_obs[:, :-1], obs_min, obs_max)
            
            obs = next_obs
            lst_action = act
            hidden = (hidden_policy, lst_action)
        
        # log p(\tau_i | m_i)
        div_term = traj_log_probs[model_indexes, np.arange(rollout_batch_size)]
        const = torch.tensor(1./num_dynamics).float().to(self.device)
        # \sum_m p(m)p(\tau_i | m)
        div_term -= torch.logsumexp(traj_log_probs + torch.log(const) , 0) # rollout_batch_size
        
        mask = ~torch.isinf(div_term)
        div_term[div_term==float("inf")] = 0
        # mi_mean = div_term.sum() / mask.sum()
        mi_loss = -div_term.sum()
        # print(f"MI {mi_mean.item()}, ratio {mask.sum() / rollout_batch_size}")
        return mi_loss 
    
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

    def eval_policy(self, number_runs):
        env = get_env(self.args['task'])
        results = ([self.test_one_trail(env) for _ in range(number_runs)])
        rewards = [result[0] for result in results]
        episode_lengths = [result[1] for result in results]
        rew_mean = np.mean(rewards)
        len_mean = np.mean(episode_lengths)
        res = OrderedDict()
        res["Eval/Reward_Mean_Env"] = rew_mean
        res["Eval/Eval_normalized_score"] = env.get_normalized_score(rew_mean)
        res["Eval/Length_Mean_Env"] = len_mean
        return res
    
    @torch.no_grad()
    def test_one_trail(self, env):
        env = deepcopy(env)
        state, done = env.reset(), False
        lst_action = torch.zeros((1,1,self.args['action_shape'])).to(self.device)
        hidden_policy = torch.zeros((1,1,self.args['lstm_hidden_unit'])).to(self.device)
        rewards = 0
        lengths = 0
        while not done:
            state = state[np.newaxis]  
            state = torch.from_numpy(state).float().to(self.device)
            hidden = (hidden_policy, lst_action)
            action, hidden_policy = self.get_meta_action(state, hidden, deterministic=True)
            use_action = action.cpu().numpy().reshape(-1)
            state_next, reward, done, _ = env.step(use_action)
            lst_action = action
            state = state_next
            rewards += reward
            lengths += 1
        return (rewards, lengths)
