from .config_base import ConfigBase


class ConfigPT:
    def __init__(self, config_general):

        env = ConfigBase()
        env.node_separation = 0.2 # this will put MPC at roughly 0.8m/s
        self.env = env

        rewards = ConfigBase()
        rewards.strategy = ['goal', 'path_advancement', 'deviation', 'deviation_end_of_path', 'corridor_hit', 'safety_corridor_raise']

        rewards.overall_scaling = 0.5 # Useful to adjust general scale of PT against HA
        rewards.params = {
            'goal': {'goal_radius': config_general.env.goal_radius, 'goal_angle': config_general.env.goal_angle, 'goal_reward': 0},
            'path_advancement': {'path_advancement_scaling': 2.0},
            'deviation': {'xy': 1, 'th': 0.05, 'scale': 0.9},
            'corridor_hit': {'penalty': -20, 'max_deviation': 1.6},
            'deviation_end_of_path': {'penalty': -10},
            'safety_corridor_raise': {'penalty': -20, 'max_deviation': 1.4},
        }

        # rewards.overall_scaling = 0.6 # Useful to adjust general scale of PT against HA
        # rewards.params = {
        #     'goal': {'goal_radius': config_general.env.goal_radius, 'goal_angle': config_general.env.goal_angle, 'goal_reward': 0},
        #     'path_advancement': {'path_advancement_scaling': 1.0},
        #     'deviation': {'xy': 1, 'th': 0.05, 'scale': 0.4},
        #     'corridor_hit': {'penalty': -10, 'max_deviation': 1.6},
        #     'deviation_end_of_path': {'penalty': -5},
        #     'safety_corridor_raise': {'penalty': -10, 'max_deviation': 1.5},
        # }
        self.rewards = rewards



        PT_model_params = {}
        MLP_local = ConfigBase()
        MLP_local.lookahead = 20
        MLP_local.lookbehind = 5
        MLP_local.node_dim = 4 # px, py, dx, dy (use of dx dy to avoid angle wrap)
        MLP_local.frame = config_general.model.frame
        MLP_local.input_size = MLP_local.node_dim * (MLP_local.lookahead + MLP_local.lookbehind)
        PT_model_params['MLP_local'] = MLP_local
        self.PT_model_params = PT_model_params
        self.PT_model_name = 'MLP_local'

        MPC = ConfigBase()
        MPC.planning_horizon = 4
        MPC.actions_in_input = 4
        self.MPC = MPC