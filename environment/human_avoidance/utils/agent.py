# original file: https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph

import numpy as np
from numpy.linalg import norm
import abc
import logging
from scripts.policy.policy_factory import policy_factory
from environment.human_avoidance.utils.action import ActionXY, ActionRot
from environment.human_avoidance.utils.state import ObservableState, FullState

from environment.path_tracking.utils import Unicycle


class Agent(object):
    SUPPORTED_KINEMATICS = ['holonomic', 'unicycle', 'unicycle_with_lag']
    def __init__(self, config, section):
        """
        Base class for robot and human. Have the physical attributes of an agent.

        """
        self.time_step = config.env.time_step
        subconfig = config.robot if section == 'robot' else config.humans
        self.visible = subconfig.visible
        self.v_max = subconfig.v_max
        self.radius = subconfig.radius
        # randomize neighbor_dist of ORCA
        if section == 'humans':
            if config.humans.randomize_attributes:
                config.orca.neighbor_dist = np.random.uniform(5, 10)
            self.policy = policy_factory[subconfig.policy](config)
            self.policy.time_step = self.time_step
        
        self.sensor_FOV = subconfig.sensor_FOV
        self.sensor_range = subconfig.sensor_range

        # for humans: we only have holonomic kinematics; for robot: depend on config_master
        self.kinematics = subconfig.kinematics
        if self.kinematics == 'unicycle_with_lag':
            self.max_lin_acc = subconfig.unicycle_with_lag_params['max_lin_acc']
            self.max_ang_acc = subconfig.unicycle_with_lag_params['max_ang_acc']
        assert self.kinematics in self.SUPPORTED_KINEMATICS
        self.px, self.py, self.gx, self.gy = None, None, None, None
        self.vx, self.vy, self.v, self.w = 0, 0, 0, 0 # all agents start at rest
        self.theta = None

    # NOTE: a lot of these functions are from the original codebase, so they may be useful for certain development tasks
    def print_info(self):
        logging.info('Agent is {} and has {} kinematic constraint'.format(
            'visible' if self.visible else 'invisible', self.kinematics))


    def sample_random_attributes(self):
        """
        Sample agent radius and v_pref attribute from certain distribution
        :return:
        """
        self.v_max = np.random.uniform(0.5, 1.5)
        self.radius = np.random.uniform(0.3, 0.5)

    def set(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        if self.kinematics == 'holonomic':
            self.vx = vx
            self.vy = vy
        else:
            # JH: I'm overloading vx and vy here to be v and r
            self.v = vx
            self.w = vy
        self.theta = theta

        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_max = v_pref

    # self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta
    def set_list(self, px, py, vx, vy, radius, gx, gy, v_pref, theta):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        if self.kinematics == 'holonomic':
            self.vx = vx
            self.vy = vy
        else:
            # JH: I'm overloading vx and vy here to be v and r
            self.v = vx
            self.w = vy
        self.theta = theta
        self.radius = radius
        self.v_max = v_pref

    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)

    def get_observable_state_list(self):
        return [self.px, self.py, self.vx, self.vy, self.radius]

    def get_observable_state_list_noV(self):
        return [self.px, self.py, self.radius]

    def get_full_state(self):
        return FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_max, self.theta)

    def get_full_state_list(self):
        return [self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_max, self.theta]

    def get_full_state_list_noV(self):
        return [self.px, self.py, self.radius, self.gx, self.gy, self.v_max, self.theta]
        # return [self.px, self.py, self.radius, self.gx, self.gy, self.v_pref]

    def get_position(self):
        return self.px, self.py
    
    def get_position_np(self):
        return np.array([self.px, self.py]) # shape: (2,)

    def get_pose_np(self):
        return np.array([self.px, self.py, self.theta]) # shape: (3,)

    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

    def get_goal_position(self):
        return self.gx, self.gy
    
    def get_goal_position_np(self):
        return np.array([self.gx, self.gy]) # shape: (2,)

    def get_velocity(self):
        return self.vx, self.vy


    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]


    @abc.abstractmethod
    def act(self, ob):
        """
        Compute state using received observation and pass it to policy

        """
        return

    def check_validity(self, action):
        if self.kinematics == 'holonomic':
            assert isinstance(action, ActionXY)
        else:
            assert isinstance(action, ActionRot)

    def step(self, action, time_step=None):
        """
        Perform an action and update the state
        """
        self.check_validity(action)
        if time_step is None:
            time_step = self.time_step
        if self.kinematics == 'holonomic':
            self.px = self.px + action.vx * time_step
            self.py = self.py + action.vy * time_step
            self.vx = action.vx
            self.vy = action.vy
        elif self.kinematics == 'unicycle':
            x = np.array([self.px, self.py, self.theta])
            u = np.array([action.v, action.w])
            y = Unicycle.step_external(x, u, time_step)
            self.theta = y[2]
            self.px, self.py = y[:2]
            self.v = action.v
            self.w = action.w

        elif self.kinematics == 'unicycle_with_lag':
            x = np.array([self.px, self.py, self.theta])
            u = np.array([action.v, action.r])

            max_change_in_v = self.max_lin_acc * self.time_step
            max_change_in_w = self.max_ang_acc * self.time_step

            diff_in_v = action.v - self.v
            v_t_plus_1 = self.v + np.sign(diff_in_v) * min(max_change_in_v, abs(diff_in_v))
            diff_in_w = action.w - self.w
            w_t_plus_1 = self.w + np.sign(diff_in_w) * min(max_change_in_w, abs(diff_in_w))
            avg_v = (self.v + v_t_plus_1) / 2
            avg_w = (self.w + w_t_plus_1) / 2
            y = Unicycle.step_external(x, np.array([avg_v, avg_w]), time_step)

            self.theta = y[2]
            self.px, self.py = y[:2]
            self.v = v_t_plus_1
            self.w = w_t_plus_1
        else:
            raise NotImplementedError

    def one_step_lookahead(self, pos, action):
        px, py = pos
        self.check_validity(action)
        new_px = px + action.vx * self.time_step
        new_py = py + action.vy * self.time_step
        new_vx = action.vx
        new_vy = action.vy
        return [new_px, new_py, new_vx, new_vy]

    def reached_destination(self):
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius

