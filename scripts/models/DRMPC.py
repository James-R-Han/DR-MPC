import torch
import torch.nn as nn
import numpy as np

from scripts.models.utils import init_, ActorLinear, CriticLinear, HumanAvoidanceNetwork, create_MLP
from torch.distributions import Categorical
from scripts.models.distributions import DiagGaussianAlphas, DiagGaussianTD3

# Sample based method of learned alpha distribution + forced diversity of learned HA actions
class DRMPC(nn.Module):
    """
    Class for the proposed network
    """
    def __init__(self, config_master, config_model):
        """
        Initializer function
        params:
        args : Training arguments
        """
        super(DRMPC, self).__init__()

        self.config_master = config_master
        self.config_model = config_model

        self.actor = ActorNetwork(config_master, config_model)
        self.shared_layers_critic1 = CriticNetwork(config_master, config_model)
        self.shared_layers_critic2 = CriticNetwork(config_master, config_model)
        additional_critic_input = 0
        self.critic1 = CriticLinear(config_master, config_model, additional_input=additional_critic_input, add_action=False)
        self.critic2 = CriticLinear(config_master, config_model, additional_input=additional_critic_input, add_action=False)

    def run_actor(self, obs, deterministic=False):
        obs_HA = obs['HA']
        obs_PT = obs['PT']
        actor_info = self.actor(obs_HA, obs_PT, deterministic=deterministic)
        
        return actor_info

    def run_critic(self, obs, action):
        obs_HA = obs['HA']
        obs_PT = obs['PT']
        
        x1 = self.shared_layers_critic1(obs_HA, obs_PT, action)
        x2 = self.shared_layers_critic2(obs_HA, obs_PT, action)
        value1 = self.critic1(x1)
        value2 = self.critic2(x2)

        return value1, value2
        

def reshapeT(T, seq_length, nenv):
    shape = T.size()[1:]
    return T.unsqueeze(0).reshape((seq_length, nenv, *shape))



class ActorNetwork(nn.Module):

    def __init__(self, config_master, config_model):

        super(ActorNetwork, self).__init__()
        self.config_master = config_master
        self.config_model = config_model
        shared_latent_space_dim = config_model.shared_latent_space

        # Hardcoded right now
        self.num_HA_actions = 6
        v_lower = config_master.config_general.robot.v_min
        v_higher = 0.8*config_master.config_general.robot.v_max # it is okay to put this number to v_max, but a nice property is that when alpha = 0 we know it is for sure focused on path tracking.
        v_middle = (v_lower + v_higher) / 2
        w_lowest = config_master.config_general.robot.w_min
        w_lower = w_lowest/3
        w_highest = config_master.config_general.robot.w_max
        w_higher = w_highest/3
        
        v_range_lower = (v_middle - v_lower)/2
        v_range_higher = (v_higher - v_middle)/2
        self.v_scaling = torch.Tensor([v_range_lower, v_range_lower, v_range_lower, v_range_higher, v_range_higher, v_range_higher]).to(config_master.config_general.device)
        v_avg_lower = (v_lower + v_middle)/2
        v_avg_higher = (v_middle + v_higher)/2
        self.v_translation = torch.Tensor([v_avg_lower, v_avg_lower, v_avg_lower, v_avg_higher, v_avg_higher, v_avg_higher]).to(config_master.config_general.device)

        w_range_low = (w_lower - w_lowest)/2
        w_range_middle = (w_higher - w_lower)/2
        w_range_high = (w_highest - w_higher)/2
        self.w_scaling = torch.Tensor([w_range_low, w_range_middle, w_range_high, w_range_low, w_range_middle, w_range_high]).to(config_master.config_general.device)
        w_avg_low = (w_lower + w_lowest)/2
        w_avg_middle = (w_lower + w_higher)/2
        w_avg_high = (w_higher + w_highest)/2
        self.w_translation = torch.Tensor([w_avg_low, w_avg_middle, w_avg_high, w_avg_low, w_avg_middle, w_avg_high]).to(config_master.config_general.device)

        self.scaling = torch.stack([self.v_scaling, self.w_scaling], dim=-1)
        self.translation = torch.stack([self.v_translation, self.w_translation], dim=-1)
        # unsqueeze
        self.scaling = torch.unsqueeze(self.scaling, 0)
        self.translation = torch.unsqueeze(self.translation, 0)

        self.HAN = HumanAvoidanceNetwork(config_master, config_model)

        input_size_to_HA_action_weight_calc = shared_latent_space_dim
        if config_model.use_time:
            input_size_to_HA_action_weight_calc += 1

        self.frame = config_model.frame

        self.HA_action_dist = create_MLP(input_size_to_HA_action_weight_calc, config_model.HA_actions_MLP, 2 * self.num_HA_actions, last_activation=False, dropout_rate=config_model.dropout_rate, use_layer_norm=config_model.use_layer_norm)
        self.dist_HA_actions = DiagGaussianTD3(config_master, self.num_HA_actions)

        self.PT_input_size = config_master.config_PT.PT_model_params[config_master.config_PT.PT_model_name].input_size
        
    
        self.PT_embedding = create_MLP(self.PT_input_size, config_model.PT_embedding_layers, shared_latent_space_dim, dropout_rate=config_model.dropout_rate, use_layer_norm=config_model.use_layer_norm)

        input_dim_to_fused_embedding = shared_latent_space_dim*2
        if config_model.use_time:
            input_dim_to_fused_embedding += 1
        
        # for past robot velocities
        input_dim_to_fused_embedding += config_master.config_general.env.lookback * config_master.config_general.action_dim

        self.fused_embedding = create_MLP(input_dim_to_fused_embedding, config_model.fusing_layers, config_model.size_of_fused_layers, dropout_rate=config_model.dropout_rate, use_layer_norm=config_model.use_layer_norm)

        self.num_MPC_actions = config_model.num_MPC_actions_to_use

        input_size_to_fusion = config_model.size_of_fused_layers + 2 * self.num_MPC_actions + self.num_HA_actions * 2 

        self.alpha_calculation = create_MLP(input_size_to_fusion, config_model.alpha_calculation_layers, self.num_HA_actions * 2, last_activation=False, dropout_rate=config_model.dropout_rate, use_layer_norm=config_model.use_layer_norm)
        self.dist_alphas = DiagGaussianAlphas(config_master)

        self.beta_adjustment = nn.Parameter(torch.ones(1)*config_model.beta_init)
        self.tanh = nn.Tanh()
        self.use_beta_adjustment = config_model.use_beta_adjustment
        
        input_size_to_prob_calc = input_size_to_fusion + self.num_HA_actions
        self.probability_calculation = create_MLP(input_size_to_prob_calc, config_model.probability_calculation_layers, self.num_HA_actions, last_activation=False, dropout_rate=config_model.dropout_rate, use_layer_norm=config_model.use_layer_norm)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

        self.features_for_OOD = None

    def forward(self, inputs_HA, inputs_PT, deterministic=False):
        MPC_action = inputs_PT['MPC_actions'][:,:2]
        MPC_actions = inputs_PT['MPC_actions'][:,:2*self.num_MPC_actions]
        HA_embedding = self.HAN(inputs_HA, inputs_PT)
        HA_action_params = self.HA_action_dist(HA_embedding)
        
        HA_action_params_flattened = HA_action_params.view(-1, 2) # this is just the mean now
        HA_actions = self.dist_HA_actions(HA_action_params_flattened, self.scaling, self.translation, deterministic=deterministic)

        # process PT
        PT_state = inputs_PT['PT_state']
        PT_embedding = self.PT_embedding(PT_state)
        
        x = torch.cat((HA_embedding, PT_embedding), dim=-1)
        
        past_robot_velocities = inputs_HA['past_robot_velocities']
        x = torch.cat((x, past_robot_velocities), dim=-1)
        x = self.fused_embedding(x)

        self.features_for_OOD = x

        input_alpha_calc = torch.cat((x, MPC_actions, HA_actions.view(-1, self.num_HA_actions*2)), dim=-1)

        alpha_distribution_params = self.alpha_calculation(input_alpha_calc)
        alpha_distribution_params_flattened = alpha_distribution_params.view(-1, 2) # for factors and MPC scaling action
        alphas_flattened, alpha_log_probs_flattened = self.dist_alphas(alpha_distribution_params_flattened, deterministic=deterministic, use_entropy_calc=False) # factors squeezed to 0 and 1 here

        alphas = alphas_flattened.view(-1, self.num_HA_actions) # 1 factor for each HA action
        # factors, MPC_scaling = factors_regrouped[:,:,0], factors_regrouped[:,:,1]
        alpha_log_probs = alpha_log_probs_flattened.view(-1, self.num_HA_actions)

        if self.use_beta_adjustment:
            beta_adjustment = torch.clamp(self.tanh(self.beta_adjustment), min=-0.5, max=0.5) # clamp so we don't over saturate. Force model to learn when alpha should be low 
            alphas = alphas + beta_adjustment
            # # bound alphas between 0 and 1
            alphas = torch.clamp(alphas, 0, 1)

        input_prob_calc = torch.cat((input_alpha_calc, alphas), dim=-1)
        probabilities = self.probability_calculation(input_prob_calc)
        probabilities = self.softmax(probabilities)

        return_info = {'alphas': alphas, 'MPC_action': MPC_action, 'HA_actions': HA_actions, 'alpha_log_probs': alpha_log_probs, 'probabilities': probabilities}
        return return_info
    

class CriticNetwork(nn.Module):
    
        def __init__(self, config_master, config_model):
    
            super(CriticNetwork, self).__init__()
    
            self.config_master = config_master
            self.config_model = config_model

            shared_latent_space_dim = config_model.shared_latent_space
    
            self.shared_network_HA = HumanAvoidanceNetwork(config_master, config_model)
    
            self.PT_input_size = config_master.config_PT.PT_model_params[config_master.config_PT.PT_model_name].input_size
            self.PT_embedding = create_MLP(self.PT_input_size, config_model.PT_embedding_layers, shared_latent_space_dim, dropout_rate=config_model.dropout_rate, use_layer_norm=config_model.use_layer_norm)
    
            input_dim_to_fused_embedding = shared_latent_space_dim*2
            if config_model.use_time:
                input_dim_to_fused_embedding += 1
                
            self.fused_embedding = create_MLP(input_dim_to_fused_embedding, config_model.fusing_layers, config_model.size_of_fused_layers, dropout_rate=config_model.dropout_rate, use_layer_norm=config_model.use_layer_norm)
            self.fused_embedding_with_action = create_MLP(config_model.size_of_fused_layers + 2, config_model.fusing_layers_with_action, config_model.size_of_fused_layers, dropout_rate=config_model.dropout_rate, use_layer_norm=config_model.use_layer_norm)


            additional_input = config_model.lookback * 2
            fused_size = config_model.size_of_fused_layers
            self.extra_MLP = create_MLP(fused_size + additional_input, [fused_size] * 2, fused_size, dropout_rate=config_model.dropout_rate, use_layer_norm=config_model.use_layer_norm, last_activation=False)

            
            self.features_for_OOD = None


        def forward(self, inputs_HA, inputs_PT, action):
            HA_embedding = self.shared_network_HA(inputs_HA, inputs_PT)
    
            # process PT
            PT_state = inputs_PT['PT_state']
            PT_embedding = self.PT_embedding(PT_state)
            
            x = torch.cat((HA_embedding, PT_embedding), dim=-1)
            x = self.fused_embedding(x)

            x = torch.cat((x, action), dim=-1)
            x = self.fused_embedding_with_action(x)

  
            past_robot_velocities = inputs_HA['past_robot_velocities']
            x = torch.cat((x, past_robot_velocities), dim=-1)
            x = self.extra_MLP(x)

            self.features_for_OOD = x

            return x
