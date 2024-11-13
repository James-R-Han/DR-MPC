from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from crowd_nav.agent.agent_b import Agent
# from crowd_sim.envs.utils.state import JointState
# from critic import DoubleQCritic
# from actor import DiagGaussianActor

class SACAgent(nn.Module):
    """SAC algorithm."""
    def __init__(self, config_master, config_SAC, model):
        super(SACAgent, self).__init__()
        
        self.config_master = config_master
        self.config_SAC = config_SAC
        self.device = torch.device(self.config_master.config_general.device)
        self.discount = self.config_SAC.discount
        self.critic_tau = self.config_SAC.critic_tau
        self.actor_update_frequency = self.config_SAC.actor_update_frequency
        self.critic_target_update_frequency = self.config_SAC.critic_target_update_frequency
        self.learnable_temperature = self.config_SAC.learnable_temperature


        self.entropy_version = self.config_SAC.entropy_version

        self.use_expected_update = self.config_SAC.use_expected_update
        self.use_avg_q = self.config_SAC.use_avg_q
        self.use_critic_clipping = self.config_SAC.use_critic_clipping
        self.init_temperature = self.config_SAC.init_temperature
        self.alpha_lr = self.config_SAC.alpha_lr_general
        self.alpha_lr_discrete = self.config_SAC.alpha_lr_discrete
        self.alpha_betas = self.config_SAC.alpha_betas
        self.model = model


        # self.critic_target = deepcopy(self.critic)
        # self.critic_target.load_state_dict(self.critic.state_dict())
        self.model_target = deepcopy(self.model)
        self.model_target.base.load_state_dict(self.model.base.state_dict())

        if self.config_SAC.policy == 'DR-MPC':
            self.num_alphas = 7
            self.has_categorical = True # First alpha will be for discrete probability distribution

            self.log_alphas = nn.ParameterList([nn.Parameter(torch.tensor(np.log(self.config_SAC.init_temperature)).to(self.device)) for _ in range(self.num_alphas)])

            self.target_entropy_all_things = self.config_SAC.target_entropy_all_things
            self.target_entropy_categorical = self.config_SAC.target_entropy_categorical
        else:
            self.log_alpha = nn.Parameter(torch.tensor(np.log(config_SAC.init_temperature)).to(self.device))
            self.log_alpha.requires_grad = True
            self.target_entropy = config_SAC.target_entropy
        self.deterministic_backup = self.config_SAC.deterministic_backup

        # optimizers
        if self.config_SAC.policy == 'DR-MPC':
            params_for_actor_optimizer = []
            for name, param in self.model.base.actor.named_parameters():
                if 'beta_adjustment' in name:
                    params_for_actor_optimizer.append({'params': param, 'lr': self.config_SAC.beta_adjustment_lr})
                else:
                    params_for_actor_optimizer.append({'params': param})
        else:
            params_for_actor_optimizer = list(self.model.base.actor.parameters()) + list(self.model.base.shared_layers_actor.parameters())
       
        self.actor_optimizer = torch.optim.AdamW(
            params_for_actor_optimizer,
            lr=self.config_SAC.actor_lr,
            betas=self.config_SAC.actor_betas,
            weight_decay=1e-4)
        self.critic_optimizer = torch.optim.AdamW(
            list(self.model.base.critic1.parameters()) +  list(self.model.base.critic2.parameters()) +  list(self.model.base.shared_layers_critic1.parameters()) + list(self.model.base.shared_layers_critic2.parameters()),
            lr=self.config_SAC.critic_lr,
            betas=self.config_SAC.critic_betas,
            weight_decay=1e-4)
        
        if self.config_SAC.policy == 'DR-MPC':
            self.log_alpha_optimizers = []
            for i in range(self.num_alphas):
                if i == 0:
                    lr_to_use = self.alpha_lr_discrete
                else:
                    lr_to_use = self.alpha_lr

                self.log_alpha_optimizers.append(torch.optim.AdamW(
                    [self.log_alphas[i]],
                    lr=lr_to_use,
                    betas=self.alpha_betas))

        else:
            self.log_alpha_optimizer = torch.optim.AdamW(
                [self.log_alpha],
                lr=self.alpha_lr,
                betas=self.alpha_betas)

        # change mode
        self.train()
        # self.critic_target.train()
        self.model_target.eval()

    def train(self, training=True):
        self.training = training
        # self.actor.train(training)
        # self.critic.train(training)
        self.model.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def alpha_tensor(self):
        alpha = torch.Tensor([alpha_i.exp() for alpha_i in self.log_alphas]).to(self.device)
        return torch.unsqueeze(alpha, 0)


    def save(self, model_dir, step):
        torch.save(self.model.state_dict(), '%s/model_%s.pt' % (model_dir, step))
        torch.save(self.model_target.state_dict(), '%s/model_target_%s.pt' % (model_dir, step))
        # also save alphas
        if self.config_SAC.policy == 'DR-MPC':

            for i in range(self.num_alphas):
                torch.save(self.log_alphas[i], '%s/log_alpha_%s_%s.pt' % (model_dir, i, step))
        else:
            torch.save(self.log_alpha, '%s/log_alpha_%s.pt' % (model_dir, step))

    def load(self, model_dir, step):

        self.model.load_state_dict(torch.load('%s/model_%s.pt' % (model_dir, step)))
        self.model_target.load_state_dict(torch.load('%s/model_target_%s.pt' % (model_dir, step)))

        # also load alphas
        if self.self.config_SAC.policy == 'DR-MPC':
            for i in range(self.num_alphas):
                loaded_alpha = torch.load('%s/log_alpha_%s_%s.pt' % (model_dir, i, step)).to(self.device)
                self.log_alphas[i] = nn.Parameter(loaded_alpha)

        else:
            self.log_alpha = torch.load('%s/log_alpha_%s.pt' % (model_dir, step))
      
        # reinitialize optimizers with loaded alpha
        if self.config_SAC.policy == 'DR-MPC':
            self.log_alpha_optimizers = []
            for i in range(self.num_alphas):
                if i == 0:
                    lr_to_use = self.alpha_lr_discrete
                else:
                    lr_to_use = self.alpha_lr

                self.log_alpha_optimizers.append(torch.optim.AdamW(
                    [self.log_alphas[i]],
                    lr=lr_to_use,
                    betas=self.alpha_betas))

        else:
            self.log_alpha_optimizer = torch.optim.AdamW(
                [self.log_alpha],
                lr=self.alpha_lr,
                betas=self.alpha_betas)
      

    def update(self, obs, action, next_obs, reward, done, step_num):

        not_done = 1 - done

        ##############################################################################################################
        # critic update
        # critic_loss, debug_value_critic = self.update_critic(obs, action, reward, next_obs, not_done)
        with torch.no_grad():
            action_Sprime_model, _, raw_log_probs_Sprime_model, info_Sprime_model = self.model.run_actor(next_obs, deterministic=False, extra_Q_info=True, model_target=self.model_target)
            
            if self.entropy_version == 'all_things':   
                sample, log_probs_Sprime_model = raw_log_probs_Sprime_model
            else:
                log_probs_Sprime_model = raw_log_probs_Sprime_model

            if self.use_expected_update:
                if self.use_avg_q:
                    min_Q_Sprime_model = info_Sprime_model['expected_avg_Q_value']
                else:
                    min_Q_Sprime_model = info_Sprime_model['expected_min_Q_value']
            else:
                Q1_Sprime_model, Q2_Sprime_model = self.model_target.run_critic(next_obs, action_Sprime_model)
                min_Q_Sprime_model = torch.min(Q1_Sprime_model, Q2_Sprime_model)
            if self.deterministic_backup:
                V_Sprime_model = torch.min(Q1_Sprime_model, Q2_Sprime_model)
            else:
                if self.entropy_version == 'all_things':
                    full_alpha_tensor = self.alpha_tensor.detach()
                    
                    # pick out relevant alphas by sample
                    if self.config_SAC.policy in ['DR-MPC']:
                        relevant_alpha_tensor_factor = torch.unsqueeze(full_alpha_tensor[0, 1:][sample], dim=1)
                        alpha_categorical = full_alpha_tensor[0, 0]
                        # repeat the alpha for the categorical action the same size of relevant factors
                        alpha_categorical_repeated = torch.ones_like(relevant_alpha_tensor_factor) * alpha_categorical
                        relevant_alpha_tensor = torch.cat([alpha_categorical_repeated, relevant_alpha_tensor_factor], dim=1)
                    else:
                        raise ValueError('entropy all_things is not a thing for non DR-MPC policies')
                
                    entropy_component = torch.sum(relevant_alpha_tensor * log_probs_Sprime_model, dim=-1, keepdim=True)

                    V_Sprime_model = min_Q_Sprime_model - entropy_component
                else:
                    V_Sprime_model = min_Q_Sprime_model - self.alpha.detach() * log_probs_Sprime_model

            target_Q_Sprime = (reward + (not_done * self.discount * V_Sprime_model)).detach()
            debug_Q = [target_Q_Sprime.mean().item(), torch.max(target_Q_Sprime).item(), torch.min(target_Q_Sprime).item()]

        # get current Q estimates
        Q1_S_taken, Q2_S_taken = self.model.run_critic(obs, action)
        if self.use_critic_clipping:
            with torch.no_grad():
                Q1_S_taken_target, Q2_S_taken_target = self.model_target.run_critic(obs, action)
            
            critic_loss1 = torch.max( (Q1_S_taken - target_Q_Sprime)**2, 
                                    ((Q1_S_taken_target + torch.clamp(Q1_S_taken - Q1_S_taken_target, min=-0.5, max=0.5)) - target_Q_Sprime)**2)
            critic_loss2 = torch.max( (Q2_S_taken - target_Q_Sprime),
                                        ((Q2_S_taken_target + torch.clamp(Q2_S_taken - Q2_S_taken_target, min=-0.5, max=0.5)) - target_Q_Sprime)**2)
            critic_loss = critic_loss1.mean() + critic_loss2.mean()
            
        else:
            critic_loss = F.mse_loss(Q1_S_taken, target_Q_Sprime) + F.mse_loss(Q2_S_taken, target_Q_Sprime)

        critic_loss_item, debug_value_critic = critic_loss.item(), debug_Q

        # execute update for critic (nothing depends on critic anymore)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.config_SAC.policy == 'DR-MPC' and self.config_SAC.reduce_beta:
            if step_num % self.config_SAC.reduce_beta_frequency == 0:
                with torch.no_grad():
                    self.model.base.actor.beta_adjustment *= self.config_SAC.reduce_beta_factor


        ##############################################################################################################
        # actor update
        if step_num % self.actor_update_frequency == 0:
            # actor_loss = self.update_actor_and_alpha(obs, step_num)
            action_S_model, _, raw_log_probs_S_model, info_S_model = self.model.run_actor(obs, deterministic=False, extra_Q_info=True)
            
            if self.entropy_version == 'all_things':     
                sample, log_probs_S_model = raw_log_probs_S_model
            else:
                log_probs_S_model = raw_log_probs_S_model
            
            if self.use_expected_update:
                min_Q_S_model = info_S_model['expected_min_Q_value']
            else:
                Q1_S_model, Q2_S_model = self.model.run_critic(obs, action_S_model)
                min_Q_S_model = torch.min(Q1_S_model, Q2_S_model)

            if self.config_SAC.use_entropy:
                if self.entropy_version == 'all_things':
                    full_alpha_tensor = self.alpha_tensor.detach()
                    if self.config_SAC.policy in ['DR-MPC']:
                        relevant_alpha_tensor_factor = torch.unsqueeze(full_alpha_tensor[0, 1:][sample], dim=1)
                        alpha_categorical = full_alpha_tensor[0, 0]
                        # repeat the alpha for the categorical action the same size of relevant factors
                        alpha_categorical_repeated = torch.ones_like(relevant_alpha_tensor_factor) * alpha_categorical
                        relevant_alpha_tensor = torch.cat([alpha_categorical_repeated, relevant_alpha_tensor_factor], dim=1)
                    else:
                        raise ValueError('entropy all_things is not a thing for non DR-MPC policies')

                    entropy_component = torch.sum(relevant_alpha_tensor * log_probs_S_model, dim=-1, keepdim=True)

                    actor_loss = (entropy_component - min_Q_S_model).mean()
                else:
                    actor_loss = (self.alpha.detach() * log_probs_S_model - min_Q_S_model).mean()
            else:
                actor_loss = (- min_Q_S_model).mean()
            
            # execute update
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            entropies = []
            if self.learnable_temperature:
                if self.config_SAC.policy == 'DR-MPC':
                    num_experience_for_all_alpha = log_probs_S_model.shape[0]

                    for i in range(len(self.log_alphas)):
                        if i == 0:
                            target_ent = self.target_entropy_categorical
                            relevant_log_probs_S_model = log_probs_S_model[:,0]
                        else:
                            target_ent = self.target_entropy_all_things
                            relevant_samples = sample == i-1
                            if torch.any(relevant_samples) != True:
                                entropies.append(0)
                                continue

                            relevant_log_probs_S_model = log_probs_S_model[relevant_samples, 1]

                        alpha_loss = (-self.log_alphas[i].exp() *(relevant_log_probs_S_model + target_ent).detach()).sum() / num_experience_for_all_alpha
                        self.log_alpha_optimizers[i].zero_grad()
                        alpha_loss.backward()
                        self.log_alpha_optimizers[i].step()

                        entropies.append(-relevant_log_probs_S_model.mean().item())
                else: 
                    self.log_alpha_optimizer.zero_grad()
                    alpha_loss = (-self.alpha *(log_probs_S_model + self.target_entropy).detach()).mean()
                    alpha_loss.backward()
                    self.log_alpha_optimizer.step()

            actor_loss_item = actor_loss.item()

        else:
            actor_loss_item = None
        ##############################################################################################################
        # target update
        if step_num % self.critic_target_update_frequency == 0:
            self.soft_update_params(self.model, self.model_target, self.critic_tau) # when we have cooling temp, want that to be updated too
        debug_values = {'debug_value_critic': debug_value_critic}
        debug_values['entropies'] = np.array(entropies)


        return critic_loss_item, actor_loss_item, debug_values


    def compute_val_loss(self, obs, action, next_obs, reward, done, step_num):
        not_done = 1 - done
        critic_loss, debug_value_critic = self.update_critic(obs, action, reward, next_obs, not_done, update=False)
        actor_loss = self.update_actor_and_alpha(obs, step_num, update=False)
        # debug_values = {'debug_value_critic': debug_value_critic}
        return critic_loss, actor_loss

    def soft_update_params(self, net, target_net, tau):
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (1 - tau)
        target_net.load_state_dict(target_net_state_dict)



