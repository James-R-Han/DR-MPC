import torch
import numpy as np

from .config_base import ConfigBase

class ConfigGeneral:
    '''
    Holds major parameters like device, model, the main environment, etc.
    Some of these parameters are needed to inform other config_master params in other files.
    '''
    def __init__(self):
        # Device settings
        use_cuda_if_available = True
        device_str = "cuda:0" if use_cuda_if_available and torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        self.seed = 1
        self.action_dim = 2
        self.save_folder = 'HA_and_PT_results'

        env = ConfigBase()
        env.time_step = 0.25
        env.time_limit = 50
        env.lookback = 6
        env.continuous_task = True
        env.goal_radius = 0.3
        env.goal_angle = 0.5
        self.env = env

        robot = ConfigBase()
        robot.action_dim = 2
        robot.v_max = 1.0
        robot.v_min = 0.0 
        assert robot.v_min == 0.0, "v_min must be 0.0"
        robot.w_max = 1.0 
        robot.w_min = -1.0 
        assert robot.w_min == -robot.w_max, "w_min must be -w_max"
        self.robot = robot




        model = ConfigBase()
        model.size = 'medium'
        if model.size == 'medium':
            self.medium_size_setup(model)
        else:
            raise NotImplementedError(f"Size '{self.model.size}' not supported.")

        SUPPORTED_POLICIES = ['DRL', 'ResidualDRL', 'DR-MPC']
        model.policy = 'DR-MPC'

        beta_adjustment_models = ['DR-MPC']
        model.has_beta = True if model.policy in beta_adjustment_models else False

        # Residual scale and additional settings
        model.residual_scale = 1.0 # 1 means that we can cover entire action space after mpc + correction action
        model.use_beta_adjustment = True
        # the more humans you have (ie. MPC path tracking is less often the optimal solution), a beta closer to 0 helps speed up learning. 
        # Even when beta_init = 0, the average alpha is 0.5 (ie. taking 50% of the MPC action)
        # It is not uncommon for the model to learn a beta_init that is not close to 0 because it learns that you can always take some of the MPC (progressing on the path).
        model.beta_init = -0.5 

        # Frame and normalization settings
        model.frame = 'robot'
        model.lookback = env.lookback
        model.use_layer_norm = True
        model.dropout_rate = None # I empirically found no noticably benefit

        # Time and PEB flags
        model.use_time = False
        model.use_PEB = True  # if we use time use_PEB should be False

        self.model = model

        OOD = ConfigBase()
        OOD.use_OOD = True
        if OOD.use_OOD:
            assert model.policy == 'DR-MPC', "OOD only supported for DR-MPC right now."
        OOD.min_dataset_size = 100
        # continuous task has smaller K because the relative number of human, robot configurations is much higher
        if env.continuous_task:
           OOD.K = 75
        else:
            OOD.K = 150

        self.OOD = OOD
        safety_module = ConfigBase
        safety_module.name = 'cvmm_diverse'
        safety_module.params = {'action_set': np.array([[0.05, -0.8], [0.15, -0.8], [0.6, -0.8], [0.05, -0.2], [0.15, -0.2], [0.6, -0.2], [0.15, 0], [0.6, 0], [0.05, 0.2], [0.15, 0.2], [0.6, 0.2], [0.05, 0.8], [0.15, 0.8],[0.6, 0.8]]), 'lookahead_steps': 8}
        self.safety_module = safety_module


        # Ensure only one of use_time or use_PEB is active
        assert self.model.use_time != self.model.use_PEB, "use_time and use_PEB cannot be both True or both False."

        # Validate policy
        if self.model.policy not in SUPPORTED_POLICIES:
            raise NotImplementedError(f"Policy '{self.model.policy}' not supported.")


    def medium_size_setup(self, config):
        ## Params for processing humans (human avoidance network)
        config.robot_embedding_size = 16

        config.HHAttn_num_heads = 8
        config.HHAttn_attn_size = 64
        config.human_embedding_size = 64
        
        # Attention and latent space settings
        config.final_attention_size = 64
        config.shared_latent_space = 64

        # Configurations for different parts of the model
        config.PT_embedding_layers = [64, 64, 64]
        config.fusing_layers = [128, 64, 64]
        config.size_of_fused_layers = 64
        config.fusing_layers_with_action = [64, 64, 64]
        config.num_MPC_actions_to_use = 4
        config.HA_actions_MLP = [32, 32, 32, 32]
        config.alpha_calculation_layers = [32, 32, 32, 32]  
        config.probability_calculation_layers = [32, 32, 32, 32]  
        
        # linear layers at end
        config.actor_layers = [64, 32, 16, 16, 8, 8]
        config.critic_layers = [64, 32, 16, 16, 8, 8, 4]

        
        
