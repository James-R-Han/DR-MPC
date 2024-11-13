import numpy as np

class RewardComputationPT:
    def __init__(self, config_PT_rewards, continuous_task):
        self.continuous_task = continuous_task
        self.config_PT_rewards = config_PT_rewards

        self.reward_strategy = self.select(config_PT_rewards.strategy)
        self.reward_params_dict = config_PT_rewards.params
    def select(self, strategy_names):
        strategies = ['goal', 'deviation', 'path_advancement',  'deviation_end_of_path', 'safety_corridor_raise', 'corridor_hit']
        reward_comp = []
        for strategy_name in strategy_names:
            if strategy_name not in strategies:
                raise ValueError(f"Strategy: '{strategy_name}' name not found.")

            reward_comp.append((strategy_name, getattr(self, strategy_name)))
        return reward_comp

    def compute_reward(self, reward_input_info={}):
        reward = 0
        done = False
        info = {'done': set(['False'])}
        for strategy_name, strategy in self.reward_strategy:
            reward_strat, done_strat = strategy(reward_input_info)
            reward_strat = self.config_PT_rewards.overall_scaling * reward_strat
            info[strategy_name] = reward_strat
            reward += reward_strat
            done = done or done_strat
            if done_strat:
                info['done'].add(strategy_name)
        return reward, done, info

    def goal(self, info):
        goal = info['goal']
        goal_radius = self.reward_params_dict['goal']["goal_radius"]
        goal_angle = self.reward_params_dict['goal']["goal_angle"]
        goal_xy = np.expand_dims(goal[:2], axis=1)
        goal_th = goal[2]

        # if any intermediate pose is within the goal radius with a good enough heading, then return goal reward
        poses = info['intermediate_poses']

        poses_xy = np.stack(poses, axis=1)[:2, :]
        poses_th = np.stack(poses, axis=1)[2, :]

        dists = np.linalg.norm(poses_xy - goal_xy, axis=0)
        angle_diff = np.abs(np.mod(poses_th - goal_th + np.pi, 2*np.pi) - np.pi)

        condition = np.logical_and(dists < goal_radius, angle_diff < goal_angle)
        if np.any(condition):
            return self.reward_params_dict['goal']["goal_reward"], True
        else:
            return 0, False


    def corridor_hit(self, info):
        pose_curr = info['pose']
        pose_on_path_curr = info['interp_pose']

        xy_dev_curr = np.linalg.norm(pose_curr[:2] - pose_on_path_curr[:2])

        if self.reward_params_dict['corridor_hit']['max_deviation'] < xy_dev_curr:
            return self.reward_params_dict['corridor_hit']['penalty'], True
        else:
            return 0, False

    def safety_corridor_raise(self, info):
        pose_curr = info['pose']
        pose_on_path_curr = info['interp_pose']

        xy_dev_curr = np.linalg.norm(pose_curr[:2] - pose_on_path_curr[:2])

        # trying this quickly
        if self.reward_params_dict['safety_corridor_raise']['max_deviation'] < xy_dev_curr:
            return self.reward_params_dict['safety_corridor_raise']['penalty'], True
        else:
            return 0, False

    def deviation_end_of_path(self, info):
        # if we are beyond the path, then we should set max deviation penalty
        prev_inter_idx = info['prev_inter_idx']
        len_path = info['len_path']
        if len_path-1 == int(prev_inter_idx):
            return self.reward_params_dict['deviation_end_of_path']['penalty'], True
        else:
            return 0, False

    def deviation(self, info):
        pose_curr = info['pose']
        pose_on_path_curr = info['interp_pose']

        xy_dev_curr = np.linalg.norm(pose_curr[:2] - pose_on_path_curr[:2])
        th_dev_curr = np.mod(pose_curr[2] - pose_on_path_curr[2] + np.pi, 2*np.pi) - np.pi

        th_reward = -self.reward_params_dict['deviation']['th'] * np.abs(th_dev_curr)
        xy_reward = -xy_dev_curr*self.reward_params_dict['deviation']['xy']
        reward = (xy_reward + th_reward)*self.reward_params_dict['deviation']['scale']
        return reward, False

    def path_advancement(self, info):
        prev_inter_idx = info['prev_inter_idx']
        curr_inter_idx = info['curr_inter_idx']

        if abs(curr_inter_idx - prev_inter_idx) > 10: # heuristic since our paths are going to be longer than 10
            # need to wrap around
            len_path = info['len_path']
            half_path = len_path//2
            prev_inter_idx_shifted = prev_inter_idx + half_path
            curr_inter_idx_shifted = curr_inter_idx + half_path
            prev_inter_idx = prev_inter_idx_shifted % len_path
            curr_inter_idx = curr_inter_idx_shifted % len_path

        reward = self.reward_params_dict['path_advancement']["path_advancement_scaling"]*(curr_inter_idx - prev_inter_idx)

        return reward, False
