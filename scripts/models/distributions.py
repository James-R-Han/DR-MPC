import numpy as np
import torch
import math
import torch.nn.functional as F

from torch import nn
from torch import distributions as pyd
from torch.distributions.transforms import AffineTransform


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0 )
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
        y = torch.clamp(y, min=-1 + 1e-15, max=1 - 1e-15)
        return torch.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale, scaling_v, translation_v, scaling_r, translation_r):
        self.loc = loc
        self.scale = scale
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.base_dist = pyd.Normal(loc, scale)
        except:
            print("stop")

        translation = torch.stack([translation_v, translation_r], dim=-1).to(self.device)
        scaling = torch.stack([scaling_v, scaling_r], dim=-1).to(self.device)
        transforms = [TanhTransform(), AffineTransform(loc=translation , scale=scaling)]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            if isinstance(tr, AffineTransform):
                mu = tr.loc + tr.scale * mu
            else:
                mu = tr(mu)
        return mu

    

class DiagGaussian(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, all_configs, log_std_bounds=[-20, 2]):
        super().__init__()
        self.all_configs = all_configs
        self.log_std_bounds = log_std_bounds

        self.v_max = all_configs.config_HA.robot.v_max
        self.w_max = all_configs.config_HA.robot.w_max

        self.outputs = dict()
       
        self.device = all_configs.config_general.device

    def forward(self, main_actions, v_base, w_base, deterministic=False, use_entropy_calc=False):
        mu, log_std = main_actions.chunk(2, dim=-1)

        log_std_min, log_std_max = self.log_std_bounds
        log_std = torch.clamp(log_std, min=log_std_min, max=log_std_max)

        std = log_std.exp()

        scaling_v = torch.ones_like(v_base)*self.v_max/2
        translation_v=self.v_max/2 - v_base # v_base has proper shape already
        scaling_w = torch.ones_like(w_base)*self.w_max 
        translation_r = -w_base # r_base has proper shape already

        scaling = torch.stack([scaling_v, scaling_w], dim=-1).to(self.device)
        translation = torch.stack([translation_v, translation_r], dim=-1).to(self.device)

        dist = pyd.Normal(mu, std)
        if deterministic:
            x = mu
            log_probs = None
        else:
            x = dist.rsample()
            if use_entropy_calc:
                raise NotImplementedError
                log_probs = -torch.sum(dist.entropy(), dim=-1, keepdim=True)
            else:
                log_probs = dist.log_prob(x).sum(axis=-1, keepdim=True)
                log_probs -= (2*(np.log(2) - x - F.softplus(-2*x))).sum(axis=1, keepdim=True)

        x = torch.tanh(x)
        x = x * scaling + translation

        return x, log_probs
    

class DiagGaussianResidualDRL(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, all_configs, log_std_bounds=[-20, 2]):
        super().__init__()
        self.all_configs = all_configs
        self.log_std_bounds = log_std_bounds

        # I'm assuming [0, vmax] and [-rmax, rmax] for regular actions
        self.v_max = all_configs.config_HA.robot.v_max
        self.w_max = all_configs.config_HA.robot.w_max

        # scaling is 1x2 tensor where scaling for v is v_max and scaling for r is r_max
        self.device = all_configs.config_general.device
        self.scaling = torch.tensor([self.v_max, self.w_max*2], device=self.device)

        self.scaling = self.scaling * all_configs.config_general.model.residual_scale

        self.outputs = dict()

    def forward(self, main_actions, deterministic=False, use_entropy_calc=False):
        mu, log_std = main_actions.chunk(2, dim=-1)

        log_std_min, log_std_max = self.log_std_bounds
        log_std = torch.clamp(log_std, min=log_std_min, max=log_std_max)

        std = log_std.exp()

        dist = pyd.Normal(mu, std)
        if deterministic:
            x = mu
            log_probs = None
        else:
            x = dist.rsample()
            if use_entropy_calc:
                raise NotImplementedError
                log_probs = -torch.sum(dist.entropy(), dim=-1, keepdim=True)
            else:
                log_probs = dist.log_prob(x).sum(axis=-1, keepdim=True)
                log_probs -= (2*(np.log(2) - x - F.softplus(-2*x))).sum(axis=1, keepdim=True)

        x = torch.tanh(x)
        x = x * self.scaling # v should be between  [-vmax, +vmax] and r should be between [-rmax, +rmax]

        return x, log_probs


class DiagGaussianAlphas(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, all_configs, log_std_bounds=[-6, 1]):
        super().__init__()
        self.all_configs = all_configs
        self.log_std_bounds = log_std_bounds

        # output is between 0 and 1 for both factors and MPC scaling
        self.scaling = 0.5
        self.translation = 0.5
       
        self.device = all_configs.config_general.device

    def forward(self, params, deterministic=False, use_entropy_calc=False):
        # params will be a tensor of shape (batch_size, 4)
        mu, log_std = params.chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        # log_std = torch.tanh(log_std)
        # log_std_min, log_std_max = self.log_std_bounds
        # log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        log_std_min, log_std_max = self.log_std_bounds
        log_std = torch.clamp(log_std, min=log_std_min, max=log_std_max)
        std = log_std.exp()

        dist = pyd.Normal(mu, std)
        if deterministic:
            x = mu
        else:
            x = dist.rsample()

        if use_entropy_calc:
            raise NotImplementedError
            log_probs = -dist.entropy()
            log_probs = torch.sum(log_probs, dim=-1, keepdim=True)
        else:
            log_probs = dist.log_prob(x).sum(axis=-1, keepdim=True)
            log_probs -= (2*(np.log(2) - x - F.softplus(-2*x))).sum(axis=1, keepdim=True)
            # log_probs = distlog_prob(x)
            # log_probs = log_probs.sum(-1, keepdim=True)
            

        x = torch.tanh(x)
        x = x * self.scaling + self.translation

        return x, log_probs
    

class DiagGaussianTD3(nn.Module):
    """fixed but decaying variance"""
    def __init__(self, all_configs, num_DO_actions):
        super().__init__()
        self.all_configs = all_configs
        self.num_DO_actions = num_DO_actions
        self.outputs = dict()
        self.device = all_configs.config_general.device

        self.fixed_std = nn.Parameter(torch.tensor([[0.15, 0.15]], device=self.device))
        self.fixed_std.requires_grad = False

        self.target_std = 0.005
        self.scaling = 0.98

    def update_fixed_std(self):
        fixed_std = torch.clamp(self.fixed_std * self.scaling, min=self.target_std)
        self.fixed_std.data = fixed_std

    def forward(self, mu, scaling, translation, deterministic=False):
        std = self.fixed_std    

        dist = pyd.Normal(mu, std)
        if deterministic:
            x = mu
        else:
            x = dist.rsample()

        x = torch.tanh(x)
        x = x.view(-1, self.num_DO_actions, 2)
        x = x * scaling + translation

        return x