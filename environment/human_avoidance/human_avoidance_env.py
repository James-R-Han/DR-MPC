import gym
import numpy as np
from numpy.linalg import norm
import copy

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import patches
from matplotlib.backends.backend_agg import FigureCanvasAgg

from environment.human_avoidance.utils.action import ActionRot, ActionXY
from environment.human_avoidance.subENVs.crowd_sim import CrowdSim
from environment.human_avoidance.utils.info import *


class HAEnv(CrowdSim):
    """
    The environment for our model with non-neural network trajectory predictors, including const vel predictor and ground truth predictor
    The number of humans at each timestep can change within a range
    """
    def __init__(self):
        super(HAEnv, self).__init__()

    def generate_ob(self):
        """Generate observation for reset and step functions"""
        ob = {}

        robot_history_flattened=np.expand_dims(self.robot_past_velocities.flatten(),0)
        # another potential frame we tried is 'path' (relative to closest path node), but yielded no advantage
        if self.frame == 'robot':
            robot_traj_in_current_frame = self.compute_position_relative_to_robot_vectorized(self.robot.px, self.robot.py, self.robot.theta, self.robot_pose_history_global[:,:2])
            # don't grab most recent position since it'll just be 0
            ob['robot_node'] =np.expand_dims(robot_traj_in_current_frame[:-1,:].flatten(),0)
        else:
            raise ValueError('frame not recognized')

        ob['past_robot_velocities'] = robot_history_flattened

        percent_into_episode = np.array([self.global_time / self.time_limit])
        # scale between -5 and 5
        scaled_percent_time_into_episode = percent_into_episode * 10 - 5 
        ob['percent_episode'] = np.expand_dims(scaled_percent_time_into_episode, 0)

        # initialize storage space for max_human_num humans
        ob['spatial_edges'] = np.ones((self.max_allowable_humans, int(2*(self.lookback+1)))) * np.inf

        if self.frame == 'robot':
            humans_relative_to_robot = self.compute_position_relative_to_robot_vectorized(self.robot.px, self.robot.py, self.robot.theta, self.human_history_global.reshape(-1,2))
            ob['spatial_edges'][:self.num_humans_in_episode] = humans_relative_to_robot.reshape(self.num_humans_in_episode, -1)
        else:
            raise ValueError('frame not recognized')

        if self.sensor_restriction_robot:
            visible_humans = self.human_valid_history != 0
            num_visible_humans = np.sum(visible_humans)

            if num_visible_humans > 0:
                ob['spatial_edges'][:num_visible_humans] = ob['spatial_edges'][visible_humans]
                ob['detected_human_num'] = np.array([num_visible_humans])
                human_valid_history = np.zeros((self.max_allowable_humans))
                human_valid_history[:num_visible_humans] = self.human_valid_history[visible_humans]
                ob['human_valid_history'] = np.expand_dims(human_valid_history, 0)
            else:
                # place a fake human far away
                ob['detected_human_num'] = np.array([1])
                ob['spatial_edges'][0, :] = -10
                human_valid_history = np.ones((self.max_allowable_humans))
                ob['human_valid_history'] = np.expand_dims(human_valid_history, 0)
        else:
            ob['detected_human_num'] = np.array([self.num_humans_in_episode])
            ob['human_valid_history'] = np.expand_dims(np.array(self.human_valid_history), 0)


        ob['spatial_edges'][np.isinf(ob['spatial_edges'])] = 15
        ob['spatial_edges'] = np.expand_dims(ob['spatial_edges'], 0)

        return ob

    def warm_start(self, num_steps):
        """
        In warmstart, robot does not move. 
        """
        for step in range(num_steps):
            human_actions = self.get_human_actions()

            # Note: This would have to change for invisible setting (which I didn't use) to make sure humans do no collide with robot
            for i, human_action in enumerate(human_actions):
                self.humans[i].step(human_action)
                newest_position = self.compute_position_relative_to_robot(self.robot.px, self.robot.py, self.robot.theta, self.humans[i].px, self.humans[i].py)
                self.human_current_velocity[i, 0], self.human_current_velocity[i, 1] = self.humans[i].vx, self.humans[i].vy

                if self.human_valid_history[i] == self.lookback+1:
                    self.human_history_relative_to_robot[i] = np.roll(self.human_history_relative_to_robot[i], -1, axis=0)
                    self.human_history_relative_to_robot[i, -1, :] = newest_position
                    self.human_history_global[i] = np.roll(self.human_history_global[i], -1, axis=0)
                    self.human_history_global[i, -1, :] = np.array([self.humans[i].px, self.humans[i].py])
                else:
                    self.human_history_relative_to_robot[i, int(self.human_valid_history[i]), :] = newest_position
                    self.human_history_global[i, int(self.human_valid_history[i]), :] = np.array([self.humans[i].px, self.humans[i].py])
                    self.human_valid_history[i] = min(self.human_valid_history[i] + 1, self.lookback+1)

                # if human is too far away, reset this human's history to 0
                if self.sensor_restriction_robot:
                    distance_between_robot_and_human = np.linalg.norm(newest_position)
                    if distance_between_robot_and_human > self.config_HA.robot.sensor_range:
                        self.human_valid_history[i] = 0

            # Update a specific human's goal once its reached its original goal
            if self.end_goal_changing:
                for i, human in enumerate(self.humans):
                    if norm((human.gx - human.px, human.gy - human.py)) < human.radius:
                        self.update_human_goal_randomly(human, 1.0)

            if self.robot_valid_history == self.lookback+1:
                self.robot_pose_history_global = np.roll(self.robot_pose_history_global, -1, axis=0)
                self.robot_pose_history_global[-1, :] = np.array([self.robot.px, self.robot.py, self.robot.theta])
                self.robot_past_velocities = np.roll(self.robot_past_velocities, -1, axis=0)
                self.robot_past_velocities[-1, :] = np.array([0,0])
            else:
                self.robot_pose_history_global[self.robot_valid_history] = np.array([self.robot.px, self.robot.py, self.robot.theta])
                self.robot_past_velocities[self.robot_valid_history-1] = np.array([0,0])
                self.robot_valid_history = min(self.robot_valid_history + 1, self.lookback+1)

        ob = self.generate_ob()
        return ob

            
    def step(self, robot_action):
        human_actions = self.get_human_actions()

        # apply action and update all agents
        self.robot.step(robot_action)
        if self.robot_valid_history == self.lookback+1:
            self.robot_pose_history_global = np.roll(self.robot_pose_history_global, -1, axis=0)
            self.robot_pose_history_global[-1, :] = np.array([self.robot.px, self.robot.py, self.robot.theta])
            self.robot_past_velocities = np.roll(self.robot_past_velocities, -1, axis=0)
            self.robot_past_velocities[-1, :] = np.array([robot_action.v, robot_action.w])
        else:
            self.robot_pose_history_global[self.robot_valid_history] = np.array([self.robot.px, self.robot.py, self.robot.theta])
            self.robot_past_velocities[self.robot_valid_history-1] = np.array([robot_action.v, robot_action.w])
            self.robot_valid_history = min(self.robot_valid_history + 1, self.lookback+1)

        
        for i, human_action in enumerate(human_actions):
            self.humans[i].step(human_action)
            
            newest_position = self.compute_position_relative_to_robot(self.robot.px, self.robot.py, self.robot.theta, self.humans[i].px, self.humans[i].py)
            
            if self.human_valid_history[i] == self.lookback+1:
                # roll and update the last element
                self.human_history_relative_to_robot[i] = np.roll(self.human_history_relative_to_robot[i], -1, axis=0)
                self.human_history_relative_to_robot[i, -1, :] = newest_position
                self.human_history_global[i] = np.roll(self.human_history_global[i], -1, axis=0)
                self.human_history_global[i, -1, :] = np.array([self.humans[i].px, self.humans[i].py])
            else:
                self.human_history_relative_to_robot[i, int(self.human_valid_history[i]), :] = newest_position
                self.human_history_global[i, int(self.human_valid_history[i]), :] = np.array([self.humans[i].px, self.humans[i].py])
                self.human_valid_history[i] = min(self.human_valid_history[i] + 1, self.lookback+1)

            # check fov here
            if self.sensor_restriction_robot:
                distance_between_robot_and_human = np.linalg.norm(newest_position)
                if distance_between_robot_and_human > self.config_HA.robot.sensor_range:
                    self.human_valid_history[i] = 0

            self.human_current_velocity[i, 0] = self.humans[i].vx
            self.human_current_velocity[i, 1] = self.humans[i].vy

        # compute reward and episode info
        self.global_time += self.time_step
        reward, done, info = self.calc_reward()


        if self.continuous_task is False or self.create_vid:
            self.global_times.append(self.global_time)
        self.step_counter = self.step_counter+1

        ob = self.generate_ob()

        # Update a specific human's goal once its reached its original goal
        if self.end_goal_changing:
            for i, human in enumerate(self.humans):
                if norm((human.gx - human.px, human.gy - human.py)) < human.radius:
                    self.update_human_goal_randomly(human, 1)
                    
        return ob, reward, done, info


    def calc_reward(self):
        done = False
        reward = 0
        info = {'done': set(['False'])}

        # timeout
        if 'timeout' in self.reward_strategy:
            if self.global_time >= self.time_limit - 1:
                reward += self.timeout_penalty
                info['timeout'] = self.timeout_penalty
                done = True
                info['done'].add('timeout')
            else:
                info['timeout'] = 0

        # collision and safety_collision
        dmin = float('inf') # dmin can be also used to add discomfort penalty if desired
        collision = False
        for i, human in enumerate(self.humans):
            dx = human.px - self.robot.px
            dy = human.py - self.robot.py
            closest_dist = (dx ** 2 + dy ** 2) ** (1 / 2) - human.radius - self.robot.radius
            if closest_dist <= 0:
                collision = True
                dmin = 0
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        # regardless if 'collision' is in self.reward_strategy, collision will terminate the episode
        if collision:
            done = True
            info['done'].add('collision')
        if 'collision' in self.reward_strategy:
            if collision:
                collision_reward = self.reward_params['collision']['penalty']
                reward += collision_reward
                info['collision'] = collision_reward
            else:
                info['collision'] = 0

        if 'safety_human_raise' in self.reward_strategy:
            if dmin < self.reward_params['safety_human_raise']['safety_dist']:
                safety_collision_reward = self.reward_params['safety_human_raise']['penalty']
                reward += safety_collision_reward
                info['safety_human_raise'] = safety_collision_reward
                done = True
                info['done'].add('safety_human_raise')
            else:
                info['safety_human_raise'] = 0


        # Careful if using sensor restriction: should set disturbance radius <= sensor range
        if 'disturbance' in self.reward_strategy:
            # Note that in the real world, we would have to wait for S'' to estimate the next vx vy (the equivalent of get_human_actions(). This is okay to do in the real world! This slight delay in credit assignment by one timestep is perfectly acceptable and does not impact the overall process.
            # TODO: could save self.get_human_actions() here for the next timestep for computation savings
            next_human_vels = self.get_human_actions() # what humans are going to do next
            disturbance_pen_v, disturbance_pen_th = 0, 0
            for human in range(self.human_num): # don't consider static humans to save some computation
                # would not hold for example in a hallway where robot and human and walking towards each other
                dx = self.robot.px - self.humans[human].px
                dy = self.robot.py - self.humans[human].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[human].radius - self.robot.radius
                if dist > self.reward_params['disturbance']['radius'] or dist <= 0:
                    continue

                curr_vel = self.human_current_velocity[human] # Note the current human vx, vy is the cvmm estimate of the human's velocity from S to S'
                fut_vel = next_human_vels[human]

                curr_v = np.linalg.norm(curr_vel)
                fut_v = np.linalg.norm(fut_vel)

                curr_th = np.arctan2(curr_vel[1], curr_vel[0])
                fut_th = np.arctan2(fut_vel[1], fut_vel[0])

                # penalize change in v and change in th
                # penalize if human has to walk slower. In real world can be absolute change, but ORCA will walk at v_pref if possible
                disturbance_pen_v_i = self.reward_params['disturbance']['factor_v'] * min(fut_v - curr_v, 0)
                th_diff = fut_th - curr_th
                th_diff = (th_diff + np.pi) % (2 * np.pi) - np.pi
                disturbance_pen_th_i = - self.reward_params['disturbance']['factor_th'] * abs(th_diff)

                scaling_factor = 1 - dist/self.reward_params['disturbance']['radius']
                disturbance_pen_v += disturbance_pen_v_i * scaling_factor
                disturbance_pen_th += disturbance_pen_th_i * scaling_factor

            reward += (disturbance_pen_v + disturbance_pen_th)
            info['disturbance_v'] = disturbance_pen_v
            info['disturbance_th'] = disturbance_pen_th

        if 'actuation_termination' in self.reward_strategy:
            past_vs = np.abs(self.robot_past_velocities[:, 0])
            total_v = np.sum(past_vs)
            if total_v < self.reward_params['actuation_termination']['min_vel']:
                actuation_termination_penalty = self.reward_params['actuation_termination']['penalty']
                reward += actuation_termination_penalty
                done = True
                info['done'].add('actuation_termination')
                info['actuation_termination'] = actuation_termination_penalty
            else:    
                info['actuation_termination'] = 0
            
        return reward, done, info       
       
    def soft_reset(self):
        ob = self.generate_ob()
        return ob

    def reset(self, path=None):
        # path information can be useful for where to put the humans. Ex. placing static human on the path

        if self.robot is None:
            raise AttributeError('robot has to be set!')
       
        self.global_time = 0
        if self.continuous_task is False:
            self.global_times = [self.global_time]
        self.step_counter = 0

        self.humans = []
        self.num_humans_in_episode = self.generate_humans(path)

        # robot state 
        self.robot_past_velocities = np.zeros((self.lookback, 2))
        # px, py, theta. Most recent position is at the end
        self.robot_pose_history_global = np.zeros((self.lookback+1, 3)) 
        self.robot_pose_history_global[:,0], self.robot_pose_history_global[:,1], self.robot_pose_history_global[:,2] = self.robot.px, self.robot.py, self.robot.theta
        self.robot_valid_history = 1

        # humans state
        self.human_history_relative_to_robot = np.zeros((self.num_humans_in_episode, self.lookback+1, 2)) # This is px,py of a human in robot frame at each different time step
        self.human_history_global = np.zeros((self.num_humans_in_episode, self.lookback+1, 2)) # This is px,py of a human in global frame. Easy to convert all to robot's current frame
        self.human_current_velocity = np.zeros((self.num_humans_in_episode, 2)) # used for reward computation
        self.human_valid_history = np.zeros((self.max_allowable_humans)) # used for sensor restriction and fluctuating human histories

        for i in range(self.num_humans_in_episode):
            self.human_history_relative_to_robot[i, 0, :] = self.compute_position_relative_to_robot(self.robot.px, self.robot.py, self.robot.theta, self.humans[i].px, self.humans[i].py)
            self.human_history_global[i, :] = np.array([self.humans[i].px, self.humans[i].py])
            self.human_current_velocity[i,0], self.human_current_velocity[i,1] = self.humans[i].vx, self.humans[i].vy

            self.human_valid_history[i] = 1
            # if human is too far away, reset this human's history to 0
            if self.sensor_restriction_robot:
                distance_between_robot_and_human = np.linalg.norm(self.human_history_relative_to_robot[i, 0, :])
                if distance_between_robot_and_human > self.config_HA.robot.sensor_range:
                    self.human_valid_history[i] = 0

        
        ob = self.generate_ob()
        return ob
