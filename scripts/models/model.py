import torch
import torch.nn as nn
import numpy as np


from scripts.models.distributions import DiagGaussian, DiagGaussianResidualDRL
from torch.distributions import Categorical
from scripts.models.DRL import DRL
from scripts.models.ResidualDRL import ResidualDRL
from scripts.models.DRMPC import DRMPC


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class BaseConfig(object):
    def __init__(self, args):
        for key, value in vars(args).items():
            setattr(self, key, value)


class Policy(nn.Module):
    """ Class for a robot policy network """
    def __init__(self, policy_name, config_master):
        super(Policy, self).__init__()
        self.base_name = policy_name
        self.config_master = config_master

        # self.convert_argparse_to_baseconfig(config_master)

        if policy_name == 'DRL':
            policy_name = DRL
        elif policy_name == 'ResidualDRL':
            policy_name = ResidualDRL
        elif policy_name == 'DR-MPC':
            policy_name = DRMPC
        else:
            raise NotImplementedError

        # cast network to torch.float32
        self.config_model = config_master.config_general.model
        self.base = policy_name(config_master, self.config_model).double()
        self.base.to(config_master.config_general.device)

        total_params = sum(p.numel() for p in self.base.parameters())
        print(f"# Model Params (actor and 2 critics): {total_params}")

        if config_master.config_HA.robot.kinematics == 'unicycle':
            self.dist_DRL = DiagGaussian(config_master)
            self.dist_ResidualDRL = DiagGaussianResidualDRL(config_master)
        else:
            raise NotImplementedError


    def run_actor(self, obs, deterministic=False, extra_Q_info=False, model_target=None):
        obs_HA, obs_PT = obs['HA'], obs['PT']
        info = {}
        actor_info = self.base.run_actor(obs, deterministic=deterministic)
        if self.base_name == 'DR-MPC':
            MPC_action,  HA_actions = actor_info['MPC_action'], actor_info['HA_actions']
            alphas, alpha_log_probs = actor_info['alphas'], actor_info['alpha_log_probs']
            probabilities = actor_info['probabilities']

            # construct all possible actions
            action_HA_ALL_with_alpha = torch.unsqueeze(alphas, dim=2) * HA_actions
            action_execute_ALL = torch.unsqueeze(MPC_action, dim=1) * (1 - torch.unsqueeze(alphas, dim=2))  + action_HA_ALL_with_alpha # shape is N x 6 x 2 (6 actions is fixed atm)
            action_execute_ALL_flattened = action_execute_ALL.view(-1, 2)

            if extra_Q_info:
                obs_HA_repeated = {k: torch.repeat_interleave(v, 6, 0) for k, v in obs_HA.items()}
                obs_PT_repeated = {k: torch.repeat_interleave(v, 6, 0) for k, v in obs_PT.items()}

                obs_repeated = {'HA': obs_HA_repeated, 'PT': obs_PT_repeated}
                if model_target is None:
                    value1_flattened, value2_flattened = self.run_critic(obs_repeated, action_execute_ALL_flattened)
                else:
                    # value1_flattened, value2_flattened, _, _ = model_target.base(obs_HA_repeated, obs_PT_repeated, action=action_execute_ALL_flattened, compute_type='critic', deterministic=deterministic)
                    value1_flattened, value2_flattened = model_target.run_critic(obs_repeated, action=action_execute_ALL_flattened)
                min_values = torch.min(value1_flattened, value2_flattened)
                min_Q_values = min_values.view(-1, 6) # Nx6
                avg_values = (value1_flattened + value2_flattened) / 2
                avg_Q_values = avg_values.view(-1, 6) # Nx6

                expected_min_Q_value = torch.sum(min_Q_values * probabilities, dim=1, keepdim=True)
                info['expected_min_Q_value'] = expected_min_Q_value
                expected_avg_Q_value = torch.sum(avg_Q_values * probabilities, dim=1, keepdim=True)
                info['expected_avg_Q_value'] = expected_avg_Q_value 

            distribution = Categorical(probs=probabilities)
            
            # get the sample and action execute based on sample
            if deterministic:
                sample = distribution.mode
            else:
                sample = distribution.sample()

            batch_size = HA_actions.size(0)
            actions_HA = HA_actions[torch.arange(batch_size), sample]
            action_execute = action_execute_ALL[torch.arange(action_execute_ALL.size(0)), sample]
            action_model = action_execute

            relevant_alphas = alphas[torch.arange(alphas.size(0)), sample]

            log_probs_of_sample = -distribution.entropy()
            log_probs_of_sample = torch.unsqueeze(log_probs_of_sample, dim=1) 
            relevant_alpha_log_probs = torch.unsqueeze(alpha_log_probs[torch.arange(batch_size), sample], dim=1)
            action_log_probs = [sample, torch.cat([log_probs_of_sample, relevant_alpha_log_probs], dim=1)]

            info['relevant_alphas'] = relevant_alphas
            info['HA_action'] = actions_HA 
            info['action_model'] = action_model

            
        elif self.base_name == 'ResidualDRL':
            MPC_action = obs_PT['MPC_actions'][:,:2]
            
            action_distribution = actor_info['action_distribution']
            action_model, action_log_probs = self.dist_ResidualDRL(action_distribution, deterministic=deterministic)

            action_execute = MPC_action + action_model
            # must clamp here
            action_execute[:, 0] = torch.clamp(action_execute[:, 0], 0.0, self.dist_ResidualDRL.v_max)
            action_execute[:, 1] = torch.clamp(action_execute[:, 1], -self.dist_ResidualDRL.w_max, self.dist_ResidualDRL.w_max)

            info['v_adjustment'] = action_model[:, 0]
            info['w_adjustment'] = action_model[:, 1]

        elif self.base_name == 'DRL':
            action_distribution = actor_info['action_distribution']
            v_base = torch.zeros(action_distribution.size(0)) # want v to be between [0, v_max]
            w_base = torch.zeros(action_distribution.size(0)) # want w to be between [-w_max, w_max] 

            action_model, action_log_probs = self.dist_DRL(action_distribution, v_base, w_base, deterministic=deterministic)
            action_execute = action_model
            
        else:
            raise NotImplementedError
        
        # make sure action_execute is always in the right range
        tolerance = 1e-5
        if (action_execute[:, 0] > self.config_master.config_general.robot.v_max + tolerance).any() or (action_execute[:, 0] < self.config_master.config_general.robot.v_min - tolerance).any():
            raise ValueError(f"Action execute v out of range: {action_execute[:, 0]}. With min: {self.config_master.config_general.robot.v_min} and max: {self.config_master.config_general.robot.v_max}.")
        if (action_execute[:, 1] > self.config_master.config_general.robot.w_max + tolerance).any() or (action_execute[:, 1] < self.config_master.config_general.robot.w_min - tolerance).any():
            raise ValueError(f"Action execute w out of range: {action_execute[:, 1]}. With min: {self.config_master.config_general.robot.w_min} and max: {self.config_master.config_general.robot.w_max}.")

        return action_model, action_execute, action_log_probs, info

    def run_critic(self, obs, action):
        value1, value2 = self.base.run_critic(obs, action)
        return value1, value2
    