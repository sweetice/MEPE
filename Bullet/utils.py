import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
ACTION_BOUND_EPSILON = 1E-6


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim), dtype='float32')
        self.action = np.zeros((max_size, action_dim), dtype='float32')
        self.next_state = np.zeros((max_size, state_dim), dtype='float32')
        self.reward = np.zeros((max_size, 1), dtype='float32')
        self.not_done = np.zeros((max_size, 1), dtype='float32')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.from_numpy(self.state[ind]).to(self.device),
            torch.from_numpy(self.action[ind]).to(self.device),
            torch.from_numpy(self.next_state[ind]).to(self.device),
            torch.from_numpy(self.reward[ind]).to(self.device),
            torch.from_numpy(self.not_done[ind]).to(self.device)
        )


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, state_dim, action_dim, hidden_dim, hidden_depth,
                 log_std_bounds):
        super().__init__()

        self.log_std_bounds = log_std_bounds

        self.outputs = dict()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)

        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.apply(weight_init)
        self.log_std_min, self.log_std_max = self.log_std_bounds

    def forward(self, obs):
        x = F.relu(self.l1(obs))
        x = F.relu(self.l2(x))

        mu, log_std = self.mu_head(x), self.log_std_head(x)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)

        # bound the log_std \in [min_std, max_std] in a soft way
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        std = log_std.exp()

        # self.outputs['mu'] = mu
        # self.outputs['std'] = std

        dist = SquashedNormal(mu, std)
        return dist



class DropoutTwinCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DropoutTwinCritic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)  # v2 from 256 -> 512
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action, mask=None):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = self.l2(q1)
        if mask is not None:
            q1 = F.relu(mask[0] * q1)
        else:
            q1 = F.relu(q1)
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        if mask is not None:
            q2 = mask[1] * q2
        q2 = self.l6(q2)
        return q1, q2
        # return q1

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class DropoutSingleCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, hidden_depth):
        super(DropoutSingleCritic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)  # v2 from 256 -> 512
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)


    def forward(self, state, action, mask=None):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = self.l2(q1)
        if mask is not None:
            q1 = F.relu(mask * q1)
        else:
            q1 = F.relu(q1)
        q1 = self.l3(q1)

        return q1

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


def init_(m):
    if isinstance(m, nn.Linear):
        gain = torch.nn.init.calculate_gain('tanh')
        torch.nn.init.orthogonal_(m.weight, gain)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def weights_init_(m):
    # weight init helper function
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
