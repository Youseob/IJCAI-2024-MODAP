import torch
from offlinerl.utils.exp import select_free_cuda

task = "Hopper-v3"
task_data_type = "low"
task_train_num = 99

# wandb_mode = "enabled"
seed = 42

device = 'cuda'+":"+str(select_free_cuda()) if torch.cuda.is_available() else 'cpu'
# device = 'cuda:0'
obs_shape = None
act_shape = None
max_action = None
# new parameters based on mopo
lstm_hidden_unit = 128
Guassain_hidden_sizes = (256,256)
value_hidden_sizes=(256,256)
hidden_sizes=(16,)
model_pool_size = 250000
rollout_batch_size = 10000
traj_num_to_infer = 1000
uniform_rollout = False

out_train_epoch = 2000
in_train_epoch = 200

train_batch_size = 256              # train policy num of trajectories

number_runs_eval = 10            # evaluation epochs in mujoco 

#-------------
dynamics_path = None
dynamics_save_path = None
only_dynamics = False

hidden_layer_size = 256
hidden_layers = 2
transition_layers = 4

transition_init_num = 23
transition_select_num = 20
transition_epoch = None
mode = 'normalize' # 'normalize', 'local', 'noRes'

real_data_ratio = 0.05
transition_batch_size = 256
policy_batch_size = 256
data_collection_per_epoch = 50e3
steps_per_epoch = 1000
max_epoch = 1000

learnable_alpha = True
transition_lr = 1e-4
actor_lr = 3e-4
critic_lr = 3e-4
discount = 0.99
soft_target_tau = 5e-3

horizon = 5
# vessl exp 2022/04/11
calibration = False
add_value_to_rt = False
belief_update_mode = 'bay' # 'bay', 'softmax', 'kl-reg'
q_lambda = 10
# temp = 10 

#tune
# params_tune = {
#     "buffer_size" : {"type" : "discrete", "value": [1e6, 2e6]},
#     "real_data_ratio" : {"type" : "discrete", "value": [0.05, 0.1, 0.2]},
#     "horzion" : {"type" : "discrete", "value": [1, 2, 5]},
#     "lam" : {"type" : "continuous", "value": [0.1, 10]},
#     "learnable_alpha" : {"type" : "discrete", "value": [True, False]},
# }

#tune
# grid_tune = {
#     "horizon" : [1, 5],
#     "lam" : [0.5, 1, 2, 5],
#     "uncertainty_mode" : ['aleatoric', 'disagreement'],
# }
