import torch
import torch.nn as nn
import numpy as np

from scripts.models.utils import init_, ActorLinear, CriticLinear, HumanAvoidanceNetwork, create_MLP


class ResidualDRL(nn.Module):
    """
    Class for the proposed network
    """
    def __init__(self, config_master, config_model):
        """
        Initializer function
        params:
        args : Training arguments
        """
        super(ResidualDRL, self).__init__()

        self.config_master = config_master
        self.config_model = config_model
        
        self.shared_layers_actor = SharedNetwork(config_master, config_model, is_critic=False)
        self.shared_layers_critic1 = SharedNetwork(config_master, config_model, is_critic=True)
        self.shared_layers_critic2 = SharedNetwork(config_master, config_model, is_critic=True)

        additional_actor_input, additional_critic_input = 0, 0

        self.actor = ActorLinear(config_master, config_model, additional_input=additional_actor_input)
        self.critic1 = CriticLinear(config_master, config_model, additional_input=additional_critic_input, add_action=False)
        self.critic2 = CriticLinear(config_master, config_model, additional_input=additional_critic_input, add_action=False)


    def run_actor(self, obs, deterministic=False):
        obs_HA = obs['HA']
        obs_PT = obs['PT']
        x = self.shared_layers_actor(obs_HA, obs_PT)
        action_distribution = self.actor(x)
        
        info = {'action_distribution': action_distribution}
        return info
    
    def run_critic(self, obs, action):
        obs_HA = obs['HA']
        obs_PT = obs['PT']
        
        x1 = self.shared_layers_critic1(obs_HA, obs_PT, action)
        x2 = self.shared_layers_critic2(obs_HA, obs_PT, action)
        value1 = self.critic1(x1)
        value2 = self.critic2(x2)

        return value1, value2

class SharedNetwork(nn.Module):

    def __init__(self, config_master, config_model, is_critic=False):

        super(SharedNetwork, self).__init__()

        self.config_master = config_master

        self.HAN = HumanAvoidanceNetwork(config_master, config_model)
        shared_latent_space_dim = config_model.shared_latent_space

        # add in path tracking MLP embedding (we are doing late fusion: after processing DO and PT, we then combine and shove it into an MLP)
        self.PT_input_size = config_master.config_PT.PT_model_params[config_master.config_PT.PT_model_name].input_size
        self.PT_input_size += 2 # +2 for MPC action

        self.PT_embedding = create_MLP(self.PT_input_size, config_model.PT_embedding_layers, shared_latent_space_dim, dropout_rate=config_model.dropout_rate, use_layer_norm=config_model.use_layer_norm)
        
        input_dim_to_fused_embedding = shared_latent_space_dim*2
        if config_model.use_time:
            input_dim_to_fused_embedding += 1
        
        input_dim_to_fused_embedding += config_model.lookback * 2 # for past velocities
        
        self.is_critic = is_critic
        if is_critic:
            input_dim_to_fused_embedding += 2

        self.fused_embedding = create_MLP(input_dim_to_fused_embedding, config_model.fusing_layers, config_model.size_of_fused_layers, dropout_rate=config_model.dropout_rate, use_layer_norm=config_model.use_layer_norm)

    def forward(self, obs_HA, obs_PT, action=None):
        HA_embedding = self.HAN(obs_HA, obs_PT)

        # process PT
        PT_state = obs_PT['PT_state']

        MPC_action = obs_PT['MPC_actions'][:,:2]
        PT_input = torch.cat((PT_state, MPC_action), dim=-1)

        PT_embedding = self.PT_embedding(PT_input)

        x = torch.cat((HA_embedding, PT_embedding), dim=-1)
        
        past_robot_velocities = obs_HA['past_robot_velocities']
        x = torch.cat((x, past_robot_velocities), dim=-1)

        if self.is_critic:
            x = torch.cat((x, action), dim=-1)
        x = self.fused_embedding(x)
        
        return x
    