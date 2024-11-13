# original file: https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph

import logging
import gym
import numpy as np
import rvo2
import random
import copy
from copy import deepcopy

from numpy.linalg import norm

from environment.human_avoidance.utils.human import Human
from environment.human_avoidance.utils.robot import Robot
from environment.human_avoidance.utils.info import *
from scripts.policy.orca import ORCA
from environment.human_avoidance.utils.state import *
from environment.human_avoidance.utils.action import ActionRot, ActionXY



class CrowdSim(gym.Env):
    """
    A base environment
    treat it as an abstract class, all other environments inherit from this one
    """
    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.
        """
        self.config_master = None
        self.config_HA = None

        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None
        self.global_times = []
        self.step_counter=0

        # reward function
        self.reward_strategy = None
        self.reward_params = None

        # simulation configuration
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None

        self.circle_radius = None
        self.human_num = None
        self.humans = []

        self.action_space=None
        self.observation_space=None

        # limit FOV
        self.robot_fov = None
        self.human_fov = None

        #seed
        self.thisSeed=None # the seed will be set when the env is created

        self.phase=None # set the phase to be train, val or test
        self.test_case=None # the test case ID, which will be used to calculate a seed to generate a human crossing case

    def configure(self, config_master, config_HA):
        """ read the config_master to the environment variables """

        self.config_master = config_master
        self.config_HA = config_HA

        self.time_limit = config_master.config_general.env.time_limit
        self.time_step = config_master.config_general.env.time_step
        self.continuous_task = config_master.config_general.env.continuous_task
        self.frame = config_master.config_general.model.frame

        self.reward_strategy = config_HA.rewards.strategy
        self.reward_params = config_HA.rewards.params

        self.circle_radius = config_HA.sim.circle_radius
        self.circle_radius_humans = config_HA.sim.circle_radius_humans
        self.human_num_base = config_HA.sim.human_num
        self.static_human_probability, self.max_static_humans = config_HA.sim.include_static_humans['episode_probability'], config_HA.sim.include_static_humans['max_static_humans']
        self.human_num_range = config_HA.sim.human_num_range
        self.max_allowable_humans = config_HA.sim.max_allowable_humans


        self.arena_size = config_HA.sim.arena_size
        self.lookback = config_HA.sim.lookback

        self.end_goal_changing = config_HA.humans.end_goal_changing

        self.sensor_restriction_robot = config_HA.robot.sensor_restriction
        
        # set robot for this envs
        rob_RL = Robot(config_HA, 'robot')
        self.set_robot(rob_RL)

    def reset(self):
        raise NotImplementedError
    
    def step(self, action):
        raise NotImplementedError

    def set_robot(self, robot):
        self.robot = robot

    def update_human_goal_randomly(self, human, chance):
        # chance variable is useful if we want to change the goal at random times
        if human.isObstacle:
            return
        if np.random.random() <= chance:
            humans_copy = []
            for h in self.humans:
                if h != human:
                    humans_copy.append(h)

            # Produce valid goal for human in case of circle setting
            while True:
                angle = np.random.random() * np.pi * 2
                gx_noise = (np.random.random() - 0.5) 
                gy_noise = (np.random.random() - 0.5)
                gx = self.circle_radius_humans * np.cos(angle) + gx_noise
                gy = self.circle_radius_humans * np.sin(angle) + gy_noise
                collide = False

                for agent in [self.robot] + humans_copy:
                    min_dist = human.radius + agent.radius + 0.2
                    if norm((gx - agent.gx, gy - agent.gy)) < min_dist or norm((gx - agent.px, gy - agent.py)) < min_dist:
                        collide = True
                        break
                if not collide:
                    break

            # Give human new goal
            human.gx = gx
            human.gy = gy
        return


    def generate_humans(self, path=None):
        """
        Set number of dynamic and static humans in the environment
        Return the number of humans in the environment
        """
        # dynamic human number
        self.human_num = np.random.randint(low=max(self.human_num_base - self.human_num_range, 1), high=self.human_num_base + self.human_num_range + 1)

        if self.config_HA.sim.scenario == 'circle_crossing':
            for i in range(self.human_num):
                self.humans.append(self.generate_circle_crossing_human(path))

            self.num_static_humans = 0
            if self.static_human_probability > 0:
                self.num_static_humans += self.generate_static_humans(path)

            for i in range(self.human_num + self.num_static_humans):
                self.humans[i].id = i

            return self.human_num + self.num_static_humans
        else:
            raise NotImplementedError


    def generate_static_humans(self, path=None):
        # TODO: Currently hardcoded based on paths for continuous task
        if self.continuous_task:
            human1 = Human(self.config_HA, 'humans')
            human1.set(-4, 0, 0, 0, 0, 0, 0)
            human2 = Human(self.config_HA, 'humans')
            human2.set(0, 0, 0, 0, 0, 0, 0)
            human1.isObstacle = True
            human2.isObstacle = True
            self.humans.append(human1)
            self.humans.append(human2)
            return 2

        # for episodic version
        probability_of_static_humans = np.random.random()
        if probability_of_static_humans > self.static_human_probability:
            return 0

        num_static_humans = np.random.randint(1, self.max_static_humans + 1)
        
        path_length = path.shape[1]
        for i in range(num_static_humans):
            human = Human(self.config_HA, 'humans')
            while True:
                random_idx = np.random.randint(0, path_length)
                px = self.PT_env.path[0, random_idx] + human.radius * 0.6 * np.random.uniform(-1, 1)
                py = self.PT_env.path[1, random_idx] + human.radius * 0.6 * np.random.uniform(-1, 1)

                # check minimum distance to start position
                dist_to_start = norm((px - self.PT_env.path[0, 0], py - self.PT_env.path[1, 0]))
                if dist_to_start < 0.8:
                    continue
                dist_to_end = norm((px - self.PT_env.path[0, -1], py - self.PT_env.path[1, -1]))
                if dist_to_end < 0.8:
                    continue

                collide = False

                min_buffer = 0
                if self.continuous_task:
                    min_buffer = 1 # don't want case where robot becomes sandwiched between two static humans and can't get out of soft-reset
                     

                for i, agent in enumerate([self.robot] + self.humans):
                    if i == 0:
                        min_dist = human.radius + agent.radius + 0.5 + min_buffer # need this on top of checking px,py from start and end goal in case we manually start robot somewhere else
                    else:
                        min_dist = human.radius + agent.radius + 0.25 + min_buffer
                    if norm((px - agent.px, py - agent.py)) < min_dist:
                        collide = True
                        break
                if not collide:
                    break

            human.set(px, py, -px, -py, 0, 0, 0) # the goal doesn't matter, it will be taking 0 action
            human.isObstacle = True
            self.humans.append(human)

        return num_static_humans

    # generate a human that starts on a circle, and its goal is on the opposite side of the circle
    def generate_circle_crossing_human(self, path=None):
        human = Human(self.config_HA, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()

        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            noise_range = 1.0
            px_noise = np.random.uniform(0, 1) * noise_range
            py_noise = np.random.uniform(0, 1) * noise_range
            px = self.circle_radius_humans * np.cos(angle) + px_noise
            py = self.circle_radius_humans * np.sin(angle) + py_noise
            collide = False

            for i, agent in enumerate([self.robot] + self.humans):
                # keep human at least 3 meters away from robot
                if i == 0:
                    min_dist = human.radius + agent.radius + 0.5
                else:
                    min_dist = human.radius + agent.radius + 0.2
                # if norm((px - agent.px, py - agent.py)) < min_dist or norm((px - agent.gx, py - agent.gy)) < min_dist:
                if norm((px - agent.px, py - agent.py)) < min_dist:
                    collide = True
                    break
            if not collide:
                break

        human.set(px, py, -px, -py, 0, 0, 0)

        return human

    def get_human_actions(self):
        # TODO: could add in FOV and sensor range restrictions here for generating human action.
        human_actions = []  # a list of all humans' actions

        for i, human in enumerate(self.humans):
            # for static humans, just append 0 action
            if human.isObstacle:
                human_actions.append(ActionXY(0, 0))
                continue

            ob = []
            for other_human in self.humans:
                if other_human != human:
                    ob.append(other_human.get_observable_state())
                    
            if self.robot.visible:
                ob += [self.robot.get_observable_state()]
                
            human_actions.append(human.act(ob))
           
        return human_actions

    def compute_position_relative_to_robot(self, robot_x, robot_y, robot_theta, pos_x, pos_y):
        cos_th = np.cos(robot_theta)
        sin_th = np.sin(robot_theta)
        C_ri = np.array([[cos_th, sin_th], [-sin_th, cos_th]])
        p_ri_i = np.array([robot_x, robot_y])
        p_hi_i = np.array([pos_x, pos_y])

        rel_vec = np.matmul(C_ri, p_hi_i - p_ri_i)
        return rel_vec

    def compute_position_relative_to_robot_vectorized(self, robot_x, robot_y, robot_theta, positions):
        # positions is a numpy array of Nx2
        cos_th = np.cos(robot_theta)
        sin_th = np.sin(robot_theta)
        C_ri = np.array([[cos_th, sin_th], [-sin_th, cos_th]])
        p_ri_i = np.expand_dims(np.array([robot_x, robot_y]), 1)

        rel_vecs = np.matmul(C_ri, np.transpose(positions) - p_ri_i)
        return np.transpose(rel_vecs)
    
    def compute_pose_relative_to_robot_vectorized(self, robot_x, robot_y, robot_theta, poses):
        positions = poses[:,:2]
        thetas = np.expand_dims(poses[:,2], 1)
        cos_th = np.cos(robot_theta)
        sin_th = np.sin(robot_theta)
        C_ri = np.array([[cos_th, sin_th], [-sin_th, cos_th]])
        p_ri_i = np.expand_dims(np.array([robot_x, robot_y]), 1)

        rel_vecs = np.matmul(C_ri, np.transpose(positions) - p_ri_i)
        rel_vecs = np.transpose(rel_vecs)
        rel_th = thetas - robot_theta          
        rel_dx = np.cos(rel_th)
        rel_dy = np.sin(rel_th)

        rel_final = np.concatenate((rel_vecs, rel_dx, rel_dy), axis=1)
        return rel_final
