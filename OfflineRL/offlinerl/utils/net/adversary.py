import torch.nn
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal
# without y
class Adversary(nn.Module):
    def __init__(self, obs_dim, num_dynamics, deterministic=False, hidden_dim=245, LOG_MAX_STD=2, LOG_MIN_STD=-20, EPS=1e-8):
        super(Adversary, self).__init__()
        self.obs_dim = obs_dim
        self.deterministic = deterministic
        self.LOG_MAX_STD = LOG_MAX_STD
        self.LOG_MIN_STD = LOG_MIN_STD
        self.EPS = EPS
        
        # input : (next_state , y, one_hot vector of dynamics )
        self.fc1 = nn.Linear( obs_dim + num_dynamics + 1, hidden_dim)
        self.fc2 = nn.Linear( hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, obs_dim)
        self.fc_logstd = nn.Linear(hidden_dim, obs_dim)

    def forward(self, inputs):
        # (bs, dim)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        log_std = self.fc_logstd(x)
        log_std = torch.clip(log_std, self.LOG_MIN_STD, self.LOG_MAX_STD)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        acts = dist.sample()
        log_prob = dist.log_prob(acts)
        return acts, log_prob 









