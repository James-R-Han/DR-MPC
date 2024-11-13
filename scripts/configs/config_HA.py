import numpy as np

from .config_base import ConfigBase

class ConfigHA:
    def __init__(self, config_general):
        # general configs for OpenAI gym env
        env = ConfigBase()
        env.warm_start = 5
        env.time_limit = 50
        env.time_step = config_general.env.time_step
        self.env = env

        # config_master for reward function
        rewards = ConfigBase()
        rewards.strategy = ['disturbance', 'collision', 'actuation_termination', 'safety_human_raise']

        if config_general.env.continuous_task is False:
            assert config_general.use_time is True
            rewards.strategy.append('timeout')

            if config_general.use_PEB:
                rewards.timeout_penalty = 0
            else:
                rewards.timeout_penalty = -5

        rewards.params = {
            'actuation_termination': {'min_vel': 0.025, 'penalty': -20},
            'safety_human_raise': {'safety_dist': 0.22, 'penalty': -15},
            'collision': {'penalty': -15},
            'disturbance': {
                'radius': 1.5,
                'factor_v': 0.8 * 7,
                'factor_th': 0.5 * 7
            }
        }
        self.rewards = rewards

        # config_master for simulation
        sim = ConfigBase()
        sim.scenario = 'circle_crossing'
        sim.circle_radius = 4
        sim.circle_radius_humans = 5
        sim.arena_size = max(sim.circle_radius, sim.circle_radius_humans) + 0.25
        sim.human_num = 6
        sim.include_static_humans = {'episode_probability': 0, 'max_static_humans': 0} # ex. for continuous task (human positions preset for now): {'episode_probability': 0, 'max_static_humans': 2 } 
        sim.human_num_range = 0


        sim.max_allowable_humans = sim.human_num + sim.human_num_range
        if sim.include_static_humans['episode_probability'] > 0:
            sim.max_allowable_humans += sim.include_static_humans['max_static_humans']

        sim.lookback = config_general.env.lookback
        sim.warm_start = True if sim.lookback > 0 else False
        self.sim = sim

        # human config_master
        humans = ConfigBase()
        humans.visible = True

        # orca or social_force for now
        humans.policy = "orca"
        humans.kinematics = "holonomic"
        if humans.policy == "orca":
            assert humans.kinematics == "holonomic"
        humans.radius = 0.3
        humans.v_max = 1.0

        humans.end_goal_changing = True
        humans.randomize_attributes = False # I haven't verified this in yet.
        humans.sensor_FOV = None # not supported right now
        humans.sensor_range = None # not supported right now


        
        self.humans = humans

        # robot config_master
        robot = ConfigBase()
        # whether robot is visible to humans (whether humans respond to the robot's motion)
        robot.visible = True # JH: personally I think this should never be False since invisible testbed defeats RL's purpose of learning how the robot's action will influence other humans
        robot.radius = 0.3
        robot.v_max = config_general.robot.v_max
        robot.v_min = config_general.robot.v_min
        robot.w_max = config_general.robot.w_max
        robot.w_min = config_general.robot.w_min
        robot.sensor_restriction = False # TODO: check that this isn't an issue...
        robot.sensor_FOV = 2 * np.pi
        robot.sensor_range = 4
        
        robot.kinematics = "unicycle" # options ['holonomic', 'unicycle', 'unicycle_with_lag']
        robot.unicycle_with_lag_params = {'max_lin_acc': 1.5, 'max_ang_acc': 2.0}

        if robot.sensor_restriction and 'disturbance' in rewards.strategy:
            assert robot.sensor_range > rewards.params['disturbance']['radius'], "Sensor range must be greater than disturbance penalty calculation radius."

        self.robot = robot

        # config_master for ORCA
        orca = ConfigBase()
        orca.neighbor_dist = 10
        orca.safety_space = 0.175
        orca.time_horizon = 5
        orca.time_horizon_obst = 5
        self.orca = orca
