import logging
import yaml
import numpy as np
# import rvo2
from numpy.linalg import norm
import torch
import matplotlib.pyplot as plt
from copy import deepcopy

# rendering related
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from matplotlib import patches
import matplotlib.lines as mlines

# PT Related
from environment.path_tracking.path_tracking_env import PTEnv
from environment.path_tracking.utils import RewardComputationPT, MPC
from environment.path_tracking.utils.unicycle import Unicycle

# DO Related
from environment.human_avoidance.human_avoidance_env import HAEnv
from environment.human_avoidance.utils.action import ActionRot, ActionXY

# configs
from scripts.configs.config_base import ConfigBase
from scripts.configs.config_PT import ConfigPT
from scripts.configs.config_HA import ConfigHA


class HAAndPTEnv:
    def __init__(self, config_master, seed_addition=0):
        self.config_master = config_master
        self.config_general = config_master.config_general
        self.config_PT = config_master.config_PT
        self.config_HA = config_master.config_HA

        self.configure_general(seed_addition=seed_addition)
        self.instantiate_MPC(self.policy)
        self.configure_PT()
        self.configure_HA()
        self.no_video()

    def instantiate_MPC(self, policy):
        self.MPC = None
        if policy in ['DR-MPC', 'ResidualDRL']:
            self.MPC =  MPC(self.config_master, self.config_PT.MPC.planning_horizon, self.time_step, self.continuous_task, self.config_general.robot.v_max, self.config_general.robot.w_max)

    def configure_general(self, seed_addition=0):
        self.policy = self.config_general.model.policy
        self.time_step = self.config_general.env.time_step
        self.use_PEB = self.config_general.model.use_PEB
        self.continuous_task = self.config_general.env.continuous_task

        # used for numpy seeding
        self.case_start = {'train': 0, 'test': np.iinfo(np.uint32).max - 3000}
        self.case_capacity = {'train': np.iinfo(np.uint32).max - 3000, 'test': 2999}
        self.case_counter = {'train': 0, 'test':0}

        # torch seeding
        torch.manual_seed(self.config_general.seed + seed_addition)
        torch.cuda.manual_seed_all(self.config_general.seed + seed_addition)

        # device
        self.device = self.config_general.device
        self.alphas = [] # fusion parameter for DR-MPC, not the entropy parameter
        self.v_adjustments = []
        self.w_adjustments = []

        policies_with_alphas = ['DR-MPC']
        self.policy_with_alphas = True if self.policy in policies_with_alphas else False

    def configure_PT(self):

        reward_computation_for_PT = RewardComputationPT(self.config_PT.rewards, self.config_general.env.continuous_task)
        self.PT_env = PTEnv(self.config_master, reward_computation_for_PT)

    def configure_HA(self):
        self.HA_env = HAEnv()
        self.HA_env.configure(self.config_master, self.config_HA)

        self.robot = self.HA_env.robot # object reference, changing one will change the other

    def configure_sim(self):
        self.sim_scenario = self.sim_params.sim_scenario
        self.sim_scenario_params = self.sim_params[self.sim_scenario]

    def create_video_continuous_task(self, vid_name, S_PT):
        self.vid_name = vid_name
        self.create_vid = True
        self.HA_env.create_vid = True
        self.HA_env.global_times = [self.HA_env.global_time]
        if self.MPC is not None:
            self.MPC_actions = [S_PT['MPC_actions'].reshape(-1,2).detach().cpu().numpy()]

        self.states = []
        self.rewards = [0]
        self.dones = [set(['False'])]
        self.infos = []
        human_state_list = [human.get_full_state() for human in self.HA_env.humans]
        self.states.append([self.robot.get_full_state(), human_state_list])

        self.alphas = []
        self.HA_actions = []
        self.actions_execute = []
        self.actions_model = []
        self.all_HA_actions = []
        self.ID = []
        self.RL_action_bool = []
        self.v_adjustments = []
        self.w_adjustments = []

        if self.continuous_task:
            self.track_path = [self.PT_env.path_idx]

    def no_video(self):
        self.create_vid = False
        self.HA_env.create_vid = False

    def soft_reset(self, done_return, info, original_Sprime):
        done = True
        consecutive_no_v = 0
        steps_in_soft_reset = 0

        Sprime = original_Sprime
        out_of_trigger_because_of_backup = False
        num_backup_steps = 0
        while done:
            # check if it's a goal or deviation end of path that accidently comes about from trying to return to a safe position
            if 'goal' in info['done'] or 'deviation_end_of_path' in info['done']:
                self.PT_env.path_idx = (self.PT_env.path_idx + 1) % len(self.PT_env.paths)
                self.PT_env.path = self.PT_env.paths[self.PT_env.path_idx]
                self.PT_env.kd_tree = self.PT_env.kd_trees[self.PT_env.path_idx]
                self.PT_env.localize_to_start = True
                if self.create_vid:
                    self.track_path.append(self.PT_env.path_idx)

                # need to generate new state
                if self.MPC is not None:
                    self.MPC.reset()
                S_PT = self.PT_env.soft_reset(self.robot.get_pose_np(), MPC=self.MPC)
                # collect MPC actions for plotting
                if self.create_vid and self.MPC is not None:
                    self.MPC_actions.append(S_PT['MPC_actions'].reshape(-1,2))
    
                S_HA = self.HA_env.soft_reset()
                
                S = {'PT': S_PT, 'HA': S_HA}

                if self.create_vid:
                    ex_info = self.infos[0]
                    zero_info = {}
                    for key in ex_info:
                        if key in ['done', 'done_PT', 'done_HA']:
                            continue
                        zero_info[key] = 0

                    self.infos.append(zero_info)

                    self.rewards.append(0)
                    self.dones.append(set(['False']))
                    human_state_list = [human.get_full_state() for human in self.HA_env.humans]
                    self.states.append([self.robot.get_full_state(), human_state_list])
                    self.HA_env.global_times.append(self.HA_env.global_time)
                    
                    if self.policy_with_alphas:
                        self.alphas.append(-2)
                    if self.policy == 'ResidualDRL':
                        self.v_adjustments.append(-2)
                        self.w_adjustments.append(-2)
                    if len(self.ID) != 0:
                        self.ID.append('NA')
                    if len(self.RL_action_bool) != 0:
                        self.RL_action_bool.append('NA')


                    self.actions_model.append(None)
                    self.actions_execute.append(None)
                    self.HA_actions.append(None)

                return S
            
            # If it is only actuation termination, then we can just return the original Sprime
            if 'actuation_termination' in info['done'] and len(info['done']) == 2: # False is always in this set
                return Sprime 

            if done_return['done_HA'] is True and consecutive_no_v > 10 and num_backup_steps < 5:
                action = ActionRot(-0.3, 0)
                num_backup_steps += 1
                out_of_trigger_because_of_backup = True
            else:
                num_backup_steps = 0
                out_of_trigger_because_of_backup = False
                curr_robot_pose = self.robot.get_pose_np()
                lookahead_path_pose = self.PT_env.query_closest_path_node_plus_one(curr_robot_pose[0], curr_robot_pose[1])
                # turn to face the lookahead_path_position
                dx = lookahead_path_pose[0] - curr_robot_pose[0]
                dy = lookahead_path_pose[1] - curr_robot_pose[1]
                desired_theta = np.arctan2(dy, dx)

                th_diff = desired_theta - curr_robot_pose[2]
                th_diff = (th_diff + np.pi) % (2 * np.pi) - np.pi
                if th_diff < 0:
                    w = -0.1
                else:
                    w = 0.1
                
                if ('corridor_hit' in info['done'] or 'safety_corridor_raise' in info['done']) and 'safety_human_raise' not in info['done']:
                    if abs(th_diff) > 0.3:
                        action = ActionRot(0, w*7)
                    else:
                        action = ActionRot(0.5, w)
                    consecutive_no_v = 0
                else:
                    action = ActionRot(0, 0)
                    consecutive_no_v += 1

            S_prime, reward_return, done_return, info, eps_done_info = self.step(action, soft_reset=True)
            
            done = done_return['done']
        
      
            steps_in_soft_reset += 1

            # this is mainly for static obstacles. Want to back up a bit more out of safety range to enable better exploration.
                # during this extra backup, want to make sure we don't hit any triggers still!
            if not done and out_of_trigger_because_of_backup:
                action = ActionRot(-0.3, 0)
                for i in range(7):
                    S_prime, reward_return, done_return, info, eps_done_info = self.step(action, soft_reset=True)
                    done = done_return['done']
                    if done:
                        break

            if steps_in_soft_reset > 250:
                self.render()
                print("Stuck in soft reset")
                return None
                # exit()
        return S_prime


    def reset(self, phase='train', test_case=None, curved_path=False):

        if test_case is None:
            case_num = self.case_start[phase] + self.case_counter[phase]
            self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_capacity[phase]
        else:
            case_num = test_case

        # position and goal generation come from numpy library
        np.random.seed(case_num)
        if self.MPC is not None:
            self.MPC.reset()

        if self.continuous_task:
            # we wil only ever 'hard reset' continuous task once.
            # Human positions will never be overriden to some value, but the robot's path will change.
            return self.reset_continuous_task()
        else:
            return self.reset_episodic_task(curved_path)

    def reset_continuous_task(self):
        self.set_robot_reset(start_angle=np.pi)
        self.prev_robot_pose = self.robot.get_pose_np()
        self.robot_change = None

        S_PT = self.PT_env.reset(self.robot.get_pose_np(), MPC=self.MPC)
     
        # collect MPC actions for plotting
        if self.create_vid and self.MPC is not None:
            self.MPC_actions = [S_PT['MPC_actions'].reshape(-1, 2)]
        
        S_HA = self.HA_env.reset(path=self.PT_env.path)
        if self.config_HA.sim.warm_start is True:
            S_HA = self.HA_env.warm_start(self.config_HA.sim.lookback)

        S = {'PT': S_PT, 'HA': S_HA}

        return S

    def reset_episodic_task(self, curved_path=None):
        raise NotImplementedError


    def set_robot_reset(self, start_angle=None):
        if self.config_HA.sim.scenario == 'circle_crossing':
            if self.robot.kinematics == 'unicycle':
                # generate robot at the bottom pointing up
                px = 0
                py = -self.config_HA.sim.circle_radius
                # py = 3
                gx = 0
                gy = self.config_HA.sim.circle_radius
                if start_angle is None:
                    start_angle = np.pi / 2
                else:
                    start_angle = start_angle
                self.robot.set(px, py, gx, gy, 0, 0, start_angle)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    
    def step(self, action, update=True, soft_reset=False, model_info=None):
        if self.continuous_task:
            return self.step_continuous_task(action, update, soft_reset, model_info)
        else:
            return self.step_episodic_task(action, update, model_info)

    def step_continuous_task(self, action, update=True, soft_reset=False, model_info=None):
        S_prime_HA, R_HA, done_HA, info_HA = self.HA_env.step(action)

        curr_robot_pose = self.robot.get_pose_np()
        self.robot_change = curr_robot_pose - self.prev_robot_pose
        
        S_prime_PT, R_PT, done_PT, info_PT = self.PT_env.step(self.prev_robot_pose, curr_robot_pose, action, MPC=self.MPC)
        if self.create_vid and self.MPC is not None:
            self.MPC_actions.append(S_prime_PT['MPC_actions'].reshape(-1,2))
        
        self.prev_robot_pose = curr_robot_pose
        reward = R_PT + R_HA
        done = done_PT or done_HA
        
        S_prime = {'PT': S_prime_PT, 'HA': S_prime_HA}
        
        union_of_info_done = info_PT['done'].union(info_HA['done'])
        info = {**info_PT, **info_HA}
        info['done'] = union_of_info_done
        info['done_PT'] = info_PT['done']
        info['done_HA'] = info_HA['done']

        
        # rendering purposes
        if self.create_vid:
            if soft_reset is False:
                if self.policy_with_alphas:
                    alpha = model_info['relevant_alphas'].item()
                    self.alphas.append(alpha)
                    HA_action = torch.squeeze(model_info['HA_action'],dim=0).detach().cpu().numpy()
                    self.HA_actions.append(HA_action)
                    action_model = torch.squeeze(model_info['action_model'],dim=0).detach().cpu().numpy()
                    self.actions_model.append(action_model)
                    action_execute = np.array([action.v, action.w])
                    self.actions_execute.append(action_execute)
                    
                if (model_info is not None and 'ID' in model_info) or len(self.ID) != 0:
                    self.ID.append(model_info['ID'])
                if (model_info is not None and 'RL_action_bool' in model_info) or len(self.RL_action_bool) != 0:
                    self.RL_action_bool.append(model_info['RL_action_bool'])
                if self.policy == 'ResidualDRL':
                    self.v_adjustments.append(model_info['v_adjustment'].item())
                    self.w_adjustments.append(model_info['w_adjustment'].item())

            else:
                if self.policy_with_alphas:
                    self.alphas.append(-1.0)
                    HA_action = self.HA_actions[-1]
                    self.HA_actions.append(HA_action)
                    action_model = self.actions_model[-1]
                    self.actions_model.append(action_model)
                    action_execute = np.array([action.v, action.w])
                    self.actions_execute.append(action_execute)

                    
                if len(self.ID) != 0:
                    self.ID.append('SOFT RESET')
                if len(self.RL_action_bool) != 0:
                    self.RL_action_bool.append('SOFT RESET')

                if self.policy == 'ResidualDRL':
                    self.v_adjustments.append(-1.0)
                    self.w_adjustments.append(-1.0)

            human_state_list = [human.get_full_state() for human in self.HA_env.humans]
            self.states.append([self.robot.get_full_state(), human_state_list])
            self.rewards.append(reward)
            self.infos.append(info)
            self.track_path.append(self.PT_env.path_idx)
            if soft_reset:
                self.dones.append(set(['SOFT RESET']))
            else:
                if done:
                    done_reasons = info_HA['done'].union(info_PT['done'])
                    done_reasons.remove('False')
                    self.dones.append(done_reasons)
                #     self.render()
                else:
                    self.dones.append(set(['False']))

        eps_done_info = {'success': False, 'collision': False, 'safety_human_raise': False, 'timeout': False, 'deviation_end_of_path': False, 'corridor_hit': False, 'safety_corridor_raise': False, 'actuation_termination': False}
        done_for_replay = False
        if done:
            done_reasons = info_HA['done'].union(info_PT['done'])
            done_reasons.remove('False')
            
            eps_done_info['success'] = 'goal' in done_reasons
            eps_done_info['collision'] ='collision' in done_reasons
            eps_done_info['safety_human_raise'] = 'safety_human_raise' in done_reasons
            eps_done_info['timeout'] =  'timeout' in done_reasons 
            eps_done_info['deviation_end_of_path'] = 'deviation_end_of_path' in done_reasons
            eps_done_info['corridor_hit'] = 'corridor_hit' in done_reasons 
            eps_done_info['safety_corridor_raise'] ='safety_corridor_raise' in done_reasons 
            eps_done_info['actuation_termination'] = 'actuation_termination' in done_reasons
            
            # want to bootstrap in the case of pure timeout (Partial-experience bootstrapping)
            if 'timeout' in done_reasons and len(done_reasons) == 1 and self.use_PEB:
                done_for_replay = False
            else:
                done_for_replay = True


        reward_return = {'R_PT': R_PT, 'R_DO': R_HA, 'R': reward}
        done_return = {'done_PT': done_PT, 'done_HA': done_HA, 'done': done, 'done_for_replay': done_for_replay}

        return S_prime, reward_return, done_return, info, eps_done_info

    def step_episodic_task(self, action, update=True, model_info=None):
        raise NotImplementedError


    def cvmm_diverse_safety_pipeline(self, action_execute, num_steps, backup_actions):
        original_action_safe, _ = self.cvmm_safety_check(action_execute, num_steps)
        info = {}
        if original_action_safe:
            safe_action = ActionRot(action_execute[0], action_execute[1])
            info['RL_action_bool'] = f'NA: ({safe_action.v}, {safe_action.w})'
            return safe_action, info
        

        dist_of_action = []
        num_safe = 0
        safe_actions = []
        safe_action_idxs = []
        for action_idx in range(backup_actions.shape[0]):
            action = backup_actions[action_idx, :]
            action_safe, dist_to_path = self.cvmm_safety_check(action, num_steps)
            dist_of_action.append(dist_to_path)
            num_safe += action_safe
            if action_safe:
                safe_actions.append(action)
                safe_action_idxs.append(action_idx)

        if original_action_safe:
            safe_actions.append(action_execute)
            safe_action_idxs.append(-1)
            num_safe += 1

        if num_safe == 0:
            # select a random action from the backup actions
            action_idx = np.random.randint(backup_actions.shape[0])
            action_to_execute = ActionRot(backup_actions[action_idx, 0], backup_actions[action_idx, 1])

            info['RL_action_bool'] = f'no safe: ({action_to_execute.v}, {action_to_execute.w})'
            return action_to_execute, info


        safe_actions = np.stack(safe_actions, axis=0)
        # grab the action with the highest velocity
        safe_action_idxs_max_speed = np.argwhere(safe_actions[:,0] == np.amax(safe_actions[:,0]))
        safe_action_idxs_max_speed = np.squeeze(safe_action_idxs_max_speed, axis=1)

        # take the action with the closest angular velocity to the original action
        target_r = action_execute[1]
        safe_actions_at_max_speed = safe_actions[safe_action_idxs_max_speed, :]
        safe_actions_r_diff = np.abs(safe_actions_at_max_speed[:,1] - target_r)
        safe_action_at_max_speed_closest_r = np.argmin(safe_actions_r_diff)
        action_idx_original = safe_action_idxs[safe_action_idxs_max_speed[safe_action_at_max_speed_closest_r]]
        safe_action = ActionRot(safe_actions_at_max_speed[safe_action_at_max_speed_closest_r, 0], safe_actions_at_max_speed[safe_action_at_max_speed_closest_r, 1])
        info['RL_action_bool'] = f"{num_safe}:{action_idx_original}:({safe_action.v},{safe_action.w})"


        return safe_action, info

    def cvmm_safety_check(self, action_execute, num_steps):
        if self.robot.kinematics != 'unicycle':
            raise NotImplementedError
        
        action = action_execute
        # current_position = self.robot.get_position_np()
        current_pose = self.robot.get_pose_np()
        
        current_humans = self.HA_env.humans
        
        FDE_to_path = 10**10
        for i in range(num_steps):
            current_pose = Unicycle.step_external(current_pose, action, self.time_step)
            current_position = current_pose[:2]
            # check for collision
            current_human_positions = []
            for human in current_humans:
                human_position = human.get_position_np() + np.array([human.vx, human.vy]) * self.time_step * (i + 1)
                current_human_positions.append(human_position)
            
            current_human_positions = np.stack(current_human_positions, axis=0)

            distance = np.linalg.norm(current_human_positions - np.expand_dims(current_position, axis=0), axis=1)
            # Note: This currently assumes that all humans are of the same radius (allows for vectorization - more important for running in real-time in the real world)
            if self.continuous_task:
                if np.any(distance < self.robot.radius + current_humans[0].radius + 0.22 + 1e-3):
                    return False, FDE_to_path
            else:
                if np.any(distance < self.robot.radius + current_humans[0].radius + 0.025):
                    return False, FDE_to_path
            
            # check for deviation (as long as not too large, we're okay)
            if i < 2:
                deviation_bool, dist_to_path = self.PT_env.query_off_path(current_position, self.config_PT.rewards.params['safety_corridor_raise']['max_deviation'])
                if deviation_bool:
                    return False, FDE_to_path

        FDE_to_path = dist_to_path
        return True, FDE_to_path

    def render(self):
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        # plot configuration
        x_offset = -0.11
        y_offset = 0.11
        text_offset=0.75
        robot_color = 'gold'
        goal_color = 'red' # 'darkgreen'
        start_color = 'darkgreen' #'gray'
        human_color = 'whitesmoke'
        human_goal_color = 'gray'
        arrow_color = 'blue' #'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        # setup base of plot
        sim_times = self.HA_env.global_times
        fig, ax = plt.subplots(figsize=(20, 14))
        buffer = 0.5
        ax.set_ylim(-self.HA_env.arena_size - buffer, self.HA_env.arena_size + buffer)
        ax.set_xlim(-self.HA_env.arena_size - buffer, self.HA_env.arena_size + buffer)
        time = plt.text(-1, self.HA_env.arena_size + buffer, 'Time: {}'.format(0), fontsize=16)
        ax.add_artist(time)
        ax.tick_params(labelsize=16)
        ax.set_xlabel('x(m)', fontsize=16)
        ax.set_ylabel('y(m)', fontsize=16)
        ax.set_aspect('equal', adjustable='box')

        # for debugging, add rewards and other useful items to plot
        done_condition = plt.text(3, self.HA_env.arena_size + buffer, 'Done: {}'.format(0), fontsize=16)
        rewards = plt.text(self.HA_env.arena_size + buffer, self.HA_env.arena_size + buffer, 'Reward: {}'.format(0), fontsize=16)
        ax.add_artist(done_condition)
        ax.add_artist(rewards)
        ex_info = self.infos[0]
        zero_info = {}
        reward_texts = []
        key_order = []
        text_index = 1
        for key in ex_info:
            if key in ['done', 'done_PT', 'done_HA']:
                continue
            key_order.append(key)
            reward_type = plt.text(self.HA_env.arena_size + buffer, self.HA_env.arena_size + buffer - text_offset * (text_index), '{}: {}'.format(key, ex_info[key]), fontsize=16)
            zero_info[key] = 0
            text_index += 1
            ax.add_artist(reward_type)
            reward_texts.append(reward_type)
        
        plot_ID = True if len(self.ID) > 0 else False
        plot_RL_action_bool = True if len(self.RL_action_bool) > 0 else False
        ID_info = self.ID + ['NA'] # this tells us if the frame we see is ID or OOD
        RL_action_bool_info = self.RL_action_bool + ['NA'] # this tells us if the RL action was executed or not
        if self.policy_with_alphas:
            alpha_info = self.alphas + [-2]
            alpha_text = plt.text(self.HA_env.arena_size + buffer, self.HA_env.arena_size + buffer - text_offset * (text_index), 'alpha: {}'.format(0), fontsize=16)
            text_index += 1
            ax.add_artist(alpha_text)

        if self.policy == 'ResidualDRL':
            v_adjustment_info = self.v_adjustments + [-2]
            v_adjustment_text = plt.text(self.HA_env.arena_size + buffer, self.HA_env.arena_size + buffer - text_offset * (text_index), 'v_adj: {}'.format(0), fontsize=16)
            ax.add_artist(v_adjustment_text)
            text_index += 1

            w_adjustment_info = self.w_adjustments + [-2]
            w_adjustment_text = plt.text(self.HA_env.arena_size + buffer, self.HA_env.arena_size + buffer - text_offset * (text_index), 'r_adj: {}'.format(0), fontsize=16)
            ax.add_artist(w_adjustment_text)
            text_index += 1

        if plot_ID:
            ID_info_text = plt.text(self.HA_env.arena_size + buffer, self.HA_env.arena_size + buffer - text_offset * (text_index), 'ID: {}'.format('NA'), fontsize=16)
            ax.add_artist(ID_info_text)
        text_index += 1
        if plot_RL_action_bool:
            RL_action_bool_text = plt.text(self.HA_env.arena_size + buffer, self.HA_env.arena_size + buffer - text_offset * (text_index), 'RL Bool: {}'.format('NA'), fontsize=16)
            ax.add_artist(RL_action_bool_text)
        # at reset, we don't know what the rewards are yet
        self.infos_for_vid = [zero_info] + self.infos

        # get and set all relevant data for update function
        robot_positions = [np.array([state[0].px, state[0].py]) for state in self.states]
        robot_states = [state[0] for state in self.states]
        human_states = [state[1] for state in self.states]

        goal = mlines.Line2D([robot_states[0].gx], [robot_states[0].gy], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')

        start = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=start_color, label='Start')
        robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=robot_color)
        ax.add_artist(start)
        ax.add_artist(goal)
        ax.add_artist(robot)
        plt.legend([robot, start, goal], ['Robot', 'Start', 'Goal'], fontsize=16, loc='upper right')

        self.actions_model.append(None)
        self.actions_execute.append(None)
        self.HA_actions.append(None)

        human_positions = [[ np.array([state[1][j].px, state[1][j].py])  for j in range(len(human_states[0]))] for state in self.states]
        humans = [plt.Circle(human_positions[0][i], human_states[0][i].radius, fill=True, facecolor=human_color, edgecolor='black')
                    for i in range(len(human_states[0]))]
        human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i),
                                    color='black', fontsize=9) for i in range(len(human_states[0]))]

      
        for i, human in enumerate(humans):
            ax.add_artist(human)
            ax.add_artist(human_numbers[i])
        
        radius = self.robot.radius
        orientations = []
        for i in range(self.HA_env.num_humans_in_episode + 1):
            orientation = []
            for state in self.states:
                if i == 0:
                    agent_state = state[0]
                    theta = agent_state.theta
                else:
                    agent_state = state[1][i - 1]
                    theta = np.arctan2(agent_state.vy, agent_state.vx)
                orientation.append(((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                        agent_state.py + radius * np.sin(theta))))
            orientations.append(orientation)
        arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)
                    for orientation in orientations]
        for arrow in arrows:
            ax.add_artist(arrow)
        global_step = -1
        offset = 0

        mpc_quiver = None
        action_execute_quiver = None
        HA_action_quiver = None
        action_model_quiver = None

        reference_path = None
        left_corridor = None
        right_corridor = None

        def update(frame_num):
            nonlocal global_step
            nonlocal arrows
            nonlocal offset
            nonlocal mpc_quiver
            nonlocal action_execute_quiver
            nonlocal HA_action_quiver
            nonlocal action_model_quiver

            nonlocal reference_path
            nonlocal left_corridor
            nonlocal right_corridor

            if self.continuous_task:
                if frame_num == 0 or self.track_path[frame_num] != self.track_path[frame_num-1]:
                    if reference_path is not None:
                        reference_path.remove()
                        left_corridor.remove()
                        right_corridor.remove()
                        reference_path, left_corridor, right_corridor = None, None, None

                    path_idx = self.track_path[frame_num]
                    path = self.PT_env.paths[path_idx]
                    th = path[2, :]
                    dx, dy = np.cos(th), np.sin(th)
                    reference_path = plt.quiver(path[0, :-1], path[1, :-1], path[0, 1:] - path[0, :-1], path[1, 1:] - path[1, :-1], width=0.005,
                            scale_units='xy', angles='xy', scale=1, label='path')

                    # plot the corridor
                    corridor_width = self.config_PT.rewards.params['corridor_hit']['max_deviation']

                    if path_idx == 0 or path_idx == 2:
                        left_corridor = plt.Circle((0, 0), 4+corridor_width, fill=False, color='black', linestyle='dotted', linewidth=1)
                        right_corridor = plt.Circle((0, 0), 4-corridor_width, fill=False, color='black', linestyle='dotted', linewidth=1)

                        ax.add_patch(left_corridor)
                        ax.add_patch(right_corridor)
                    else:
                        # straight lines at x = -corridor_width and x = corridor_width
                        left_corridor = mlines.Line2D([-corridor_width, -corridor_width], [-4, 4], color='black', linestyle='dotted', linewidth=1)
                        right_corridor = mlines.Line2D([corridor_width, corridor_width], [-4, 4], color='black', linestyle='dotted', linewidth=1)
                        ax.add_artist(left_corridor)
                        ax.add_artist(right_corridor)
            
            global_step += 1
            robot.center = robot_positions[frame_num]
            goal.set_data([robot_states[frame_num].gx], [robot_states[frame_num].gy])
            for i, human in enumerate(humans):
                human.center = human_positions[frame_num][i]
                human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] - y_offset))
                for arrow in arrows:
                    arrow.remove()
                arrows = [patches.FancyArrowPatch(*orientation[frame_num], color=arrow_color,
                                                    arrowstyle=arrow_style) for orientation in orientations]
                for arrow in arrows:
                    ax.add_artist(arrow)
          
            # time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))
            time.set_text('Time: {:.2f}'.format(sim_times[frame_num]))
            rewards.set_text('Reward: {:.2f}'.format(self.rewards[frame_num]))
            done_condition_text = ''
            if 'False' in self.dones[frame_num]:
                done_condition_text = 'False'
            else:
                for condition in self.dones[frame_num]:
                    done_condition_text += condition + ', '
            done_condition.set_text('Done: {}'.format(done_condition_text))
            for i, key in enumerate(key_order):
                reward_texts[i].set_text('{}: {:.2f}'.format(key, self.infos_for_vid[frame_num][key]))

            if self.policy_with_alphas:
                alpha_text.set_text('alpha: {:.3f}'.format(alpha_info[frame_num]) )

            if self.policy == 'ResidualDRL':
                v_adjustment_text.set_text('v_adj: {:.3f}'.format(v_adjustment_info[frame_num]))
                w_adjustment_text.set_text('w_adj: {:.3f}'.format(w_adjustment_info[frame_num]))

            if self.MPC is not None:
                if mpc_quiver is not None:
                    mpc_quiver.remove()
                MPC_actions = self.MPC_actions[frame_num]
                curr_robot_state = robot_states[frame_num]
                prev_pose = np.array([curr_robot_state.px, curr_robot_state.py, curr_robot_state.theta])
                num_actions = MPC_actions.shape[0]
                future_poses = np.zeros((3, num_actions))
                for action_num in range(num_actions):
                    action = MPC_actions[action_num,:]
                    prev_pose = Unicycle.step_external(prev_pose, action, self.time_step)
                    future_poses[:, action_num] = prev_pose

                th_future_poses = future_poses[2, :]
                dx_future_poses = np.cos(th_future_poses)
                dy_future_poses = np.sin(th_future_poses)
                mpc_quiver = plt.quiver(future_poses[0, :], future_poses[1, :], dx_future_poses, dy_future_poses, label='future poses', color='y')

            if self.policy_with_alphas:
                if action_execute_quiver is not None:
                    action_execute_quiver.remove()
                    action_execute_quiver = None
                if HA_action_quiver is not None:
                    HA_action_quiver.remove()
                    HA_action_quiver = None
                if action_model_quiver is not None:
                    action_model_quiver.remove()
                    action_model_quiver = None

                curr_robot_state = robot_states[frame_num]
                prev_pose = np.array([curr_robot_state.px, curr_robot_state.py, curr_robot_state.theta])

                try:
                    action_execute = self.actions_execute[frame_num]
                    DO_action = self.HA_actions[frame_num]
                    pose_from_action_execute = Unicycle.step_external(prev_pose, action_execute, self.time_step)
                    pose_from_action_DO = Unicycle.step_external(prev_pose, DO_action, self.time_step)

                    th_action_execute = pose_from_action_execute[2]
                    dx_action_execute, dy_action_execute = np.cos(th_action_execute), np.sin(th_action_execute)
                    th_DO_action = pose_from_action_DO[2]
                    dx_DO_action, dy_DO_action = np.cos(th_DO_action), np.sin(th_DO_action)
                    
                    
                    action_model = self.actions_model[frame_num]
                    pose_from_action_model = Unicycle.step_external(prev_pose, action_model, self.time_step)
                    th_action_model = pose_from_action_model[2]
                    dx_action_model, dy_action_model = np.cos(th_action_model), np.sin(th_action_model)
                    
                    HA_action_quiver = plt.quiver(pose_from_action_DO[0], pose_from_action_DO[1], dx_DO_action, dy_DO_action, label='DO action', color='orange')
                    action_model_quiver = plt.quiver(pose_from_action_model[0], pose_from_action_model[1], dx_action_model, dy_action_model, label='action model', color='green')
                    action_execute_quiver = plt.quiver(pose_from_action_execute[0], pose_from_action_execute[1], dx_action_execute, dy_action_execute, label='action execute', color='r')
                except:
                    pass # we will have None values when in soft reset, so we can just ignore and not plot them

            if plot_ID:
                ID_info_text.set_text('ID: {}'.format(ID_info[frame_num]))

            if plot_RL_action_bool:
                RL_action_bool_text.set_text('RL Bool: {}'.format(RL_action_bool_info[frame_num]))

        frames_length = len(self.dones)
        anim = animation.FuncAnimation(fig, update, frames=frames_length, interval=int(round(self.HA_env.time_step * 1000.0)))
        anim.running = True
        ffmpeg_writer = animation.writers['ffmpeg']
        writer = ffmpeg_writer(fps=int(round(1.0 / self.HA_env.time_step)) * 2, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(self.vid_name, writer=writer)
        plt.close(fig)
