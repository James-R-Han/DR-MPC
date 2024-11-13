from scripts.configs.config_HA import ConfigHA
from scripts.configs.config_base import ConfigBase

class ConfigSAC:

    def __init__(self, config_general):
        if config_general.env.continuous_task:
            self.discount = 0.95
        else:
            self.discount = 0.99

        self.policy = config_general.model.policy

        # a list structure makes it easier for prototyping and adding new models
        if self.policy in ['DR-MPC']:
            self.entropy_version = 'all_things'
            self.target_entropy_all_things = -1.5
            self.target_entropy_categorical = 0.05
            self.use_expected_update = True
            self.use_avg_q = True
            self.use_critic_clipping = True

        else:
            # for DRL and DRL_MPC_Residual models
            self.entropy_version = 'regular'
            self.target_entropy = -2.0
            self.use_expected_update = False
            self.use_avg_q = False
            self.use_critic_clipping = False

        self.use_entropy = True
        self.deterministic_backup = False
        self.learnable_temperature = True
        self.init_temperature = 0.1
        self.alpha_lr_general = 2e-5
        self.alpha_lr_discrete = 1e-5

        self.alpha_betas = [0.9, 0.999]
        self.actor_lr = 1e-4
        self.actor_betas = [0.9, 0.999]
        self.actor_update_frequency = 1
        self.critic_lr = 1e-4
        self.critic_betas = [0.9, 0.999]
        self.critic_tau = 0.005
        self.critic_target_update_frequency = 1

        self.reduce_beta = True # not necessary
        self.beta_adjustment_lr = 3e-5
        self.reduce_beta_frequency = 100
        self.reduce_beta_factor = 0.99

