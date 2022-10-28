import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import DiagGaussianActor, DropoutSingleCritic
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ME_SAC:
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 discount=0.99,
                 tau=0.005,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_freq=2,

                 # SAC  parameters
                 init_temperature = 0.1,
                 alpha_lr = 1e-4,
                 learn_alpha = True,
                 dropout_p = 0.1
                 ):
        super(ME_SAC, self).__init__()
        self.actor = DiagGaussianActor(state_dim, action_dim, hidden_dim=256, hidden_depth=2,
                                       log_std_bounds=[-20, 2]).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = DropoutSingleCritic(state_dim, action_dim, hidden_dim=256, hidden_depth=2).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.learn_alpha = learn_alpha
        self.log_alpha = torch.FloatTensor([np.log(init_temperature)]).to(device).requires_grad_(True)
        self.target_entropy = - action_dim

        if self.learn_alpha:
            self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.dropout = nn.Dropout(dropout_p).to(device)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

        # Dropout parameters
        self.dim_hidden_layer = 256
        self.ones = torch.ones(256, 256).to(device)

    def select_action(self, state, sample=False):
        with torch.no_grad():
            state = torch.from_numpy(state).reshape(1, -1).to(device)
            dist = self.actor(state)
            action = dist.sample() if sample else dist.mean
            action = action.clamp(-self.max_action, self.max_action)
            return action.cpu().numpy().flatten()


    def train(self, replay_buffer, batch_size):

        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Compute critic loss
        with torch.no_grad():
            next_action_dist = self.actor(next_state)
            next_action = next_action_dist.rsample()
            next_action_log_prob = next_action_dist.log_prob(next_action).sum(-1, keepdim=True)  # TODO: Figureout this line
            # ones = torch.ones(batch_size, self.dim_hidden_layer).to(device)
            drop = self.dropout(self.ones)
            target_V = self.critic_target(next_state, next_action, drop) - self.alpha.detach() * next_action_log_prob
            target_Q = reward + not_done * self.discount * target_V

        current_Q = self.critic(state, action, drop)
        critic_loss = F.mse_loss(current_Q, target_Q)

        # learn critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss and alpha loss
        current_action_dist = self.actor(state)
        current_action = current_action_dist.rsample()
        current_action_log_prob = current_action_dist.log_prob(current_action).sum(-1, keepdim=True)
        current_action_Q = self.critic(state, current_action)
        actor_loss = (self.alpha.detach() * current_action_log_prob - current_action_Q).mean()

        if self.learn_alpha:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-current_action_log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
        else:
            alpha_loss = 0

        if self.total_it % self.policy_freq ==0:

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.total_it += 1

    @property
    def alpha(self):
        return self.log_alpha.exp()
