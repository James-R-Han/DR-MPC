
import numpy as np

from environment.path_tracking.utils.path_tracking_core import PTEnvCore
from environment.path_tracking.utils import Unicycle

class PTEnv(PTEnvCore):
    def __init__(self, config_master, reward_comp):
        self.config_master = config_master
        self.config_PT = config_master.config_PT

        self.RewardComp = reward_comp
        self.PT_model_name = self.config_PT.PT_model_name
        self.model_kwargs = self.config_PT.PT_model_params[self.PT_model_name]
        self.time_step = self.config_master.config_general.env.time_step

        self.MPC_elements_in_input = self.config_PT.MPC.actions_in_input * self.config_master.config_general.action_dim
        self.continuous_task = self.config_master.config_general.env.continuous_task

        self.localize_to_start = True

    def reset(self, start_pose, end_loc=None, curr_angle=None, MPC=None, curved_path=False):
        if self.continuous_task:
            return self.reset_continuous_task(start_pose, MPC)
        else:
            return self.reset_episodic_task(start_pose, end_loc, curr_angle, MPC, curved_path)

    def reset_continuous_task(self, start_pose, MPC=None):

        radius = abs(start_pose[1])

        bottom_pose = np.array([0, -radius, np.pi])
        path0 = self.generate_circle_path(start_pose)
        ktime_stepree0 = self.build_kdtree(path0)

        path1 = self.generate_direct_path(np.array([0,-radius,np.pi/2]), np.array([0,radius,np.pi/2]))
        ktime_stepree1 =self.build_kdtree(path1)

        top_pose = np.array([0,radius,np.pi])
        path2 = self.generate_circle_path(top_pose, CCW=True)
        ktime_stepree2 =self.build_kdtree(path2)

        path3 = self.generate_direct_path(np.array([0,radius,-np.pi/2]), np.array([0,-radius,-np.pi/2]))
        ktime_stepree3 =self.build_kdtree(path3)

        self.paths = [path0, path1, path2, path3]
        self.kd_trees = [ktime_stepree0, ktime_stepree1, ktime_stepree2, ktime_stepree3]

        self.path_idx = 0
        self.path = path0
        self.kd_tree = ktime_stepree0

        self.localize_to_start = True
        state_PT = self.generate_state(self.path, start_pose, self.kd_tree, MPC=MPC)

        return state_PT

    def soft_reset(self, curr_pose, MPC=None):
        state_PT = self.generate_state(self.path, curr_pose, self.kd_tree, MPC)
        return state_PT

    def reset_episodic_task(self, start_pose, end_loc, curr_angle, MPC=None, curved_path=False):
        start_loc = start_pose[:2]
        if curved_path:
            self.path = self.generate_curved_path(start_loc, end_loc)
        else:
            self.path = self.generate_direct_path(start_loc, end_loc)
        
        self.kd_tree = self.build_kdtree(self.path)

        curr_pose = np.array([start_loc[0], start_loc[1], curr_angle])
        state_PT = self.generate_state(self.path, curr_pose, self.kd_tree, MPC=MPC)
        return state_PT


    def set_path(self, path):
        self.path = path
        self.kd_tree =self.build_kdtree(self.path)

    def step(self, prev_pose, curr_pose, action, MPC=None):
        intermediate_poses = []
        for i in range(1, 6):
            pose_i = Unicycle.step_external(prev_pose, action, self.time_step*i/5)
            intermediate_poses.append(pose_i)

        prev_interp_pose, prev_interp_index, prev_closest_indices = self.localize_on_path(self.kd_tree, prev_pose[:2], self.path,  strict=False, localize_to_start=self.localize_to_start)
        curr_interp_pose, curr_interp_index, curr_closest_indices = self.localize_on_path(self.kd_tree, curr_pose[:2], self.path, strict=False, localize_to_start=self.localize_to_start)
        
        if curr_interp_index >= self.path.shape[1]//3 and curr_interp_index < 2*self.path.shape[1]//3:
            self.localize_to_start = False


        return_state = self.generate_state(self.path, curr_pose, self.kd_tree, MPC=MPC, interp_pose=curr_interp_pose, interp_index=curr_interp_index)

        len_path = self.path.shape[1]
        reward_comp_input = {'pose_before': prev_pose, 'pose': curr_pose, 'intermediate_poses': intermediate_poses, 'action':action,
                            'goal': self.path[:,-1], 'prev_inter_idx': prev_interp_index, 'curr_inter_idx': curr_interp_index,
                            'interp_pose': curr_interp_pose, 'interp_pose_before': prev_interp_pose, 'len_path': len_path}
        reward, done, reward_info = self.RewardComp.compute_reward(reward_comp_input)

        return return_state, reward, done, reward_info

    def generate_circle_path(self, start_pose, CCW=False):
        robot = Unicycle(time_step=self.time_step, x0=start_pose)
        radius = abs(start_pose[1])
        v = self.config_PT.env.node_separation/self.time_step
        
        num_steps = 0
        target_diameter = radius*2
        w = -v/target_diameter *2
        if CCW:
            w = -w
        
        w_abs = np.abs(w)

        num_steps = int(np.pi * 2 / w_abs / self.time_step) - 1
        action_for_circle = np.array([v, w])

        for i in range(num_steps):
            robot.step(action_for_circle)
            
        return robot.history

    def generate_curved_path(self, start_loc, end_loc):
        angle = np.arctan2(end_loc[1] - start_loc[1], end_loc[0] - start_loc[0])
        start_pose = np.array([start_loc[0], start_loc[1], angle])

        dist = np.linalg.norm(end_loc - start_loc)
        node_separation = self.PT_CS_ENV_params.env.node_separation
        num_nodes = int(dist / node_separation) + 1

        robot = Unicycle(time_step=self.time_step, x0=start_pose)
        v = self.node_separation/self.time_step
        
        num_steps = 0

        # random start
        u_random = np.array([v, np.random.uniform(-1, 1)])
        for i in range(6):
            robot.step(u_random)
            num_steps += 1
        
        # try to correct back onto "middle" path
        while num_steps < num_nodes:
            x_coord = robot.x[0,0]
            th = robot.x[2,0]
            th  = (th+np.pi)%(2*np.pi) - np.pi 

            tolerance = 20*np.pi/180

            # if on left, go right. If on right, go left. BUT, want theta between 0 and pi (ie. pointing up)
            if x_coord < 0:
                # on left, want theta between 0 and pi/2
                proposed_w = np.random.uniform(-1, 0) # turning right means decreasing rads
                future_th = th + proposed_w*self.time_step

                if future_th < 0 + tolerance:
                    u = np.array([v, 0])
                else:
                    u = np.array([v, proposed_w])
            else:
                # on right, want theta between pi/2 and pi
                proposed_w = np.random.uniform(0, 1) # turning left means increasing rads
                future_th = th + proposed_w*self.time_step

                if future_th > np.pi - tolerance:
                    u = np.array([v, 0])
                else:
                    u = np.array([v, proposed_w])

            robot.step(u)
            num_steps += 1

        return robot.history

    def generate_direct_path(self, start_loc, end_loc):
        angle = np.arctan2(end_loc[1] - start_loc[1], end_loc[0] - start_loc[0])
        start_pose = np.array([start_loc[0], start_loc[1], angle])
        end_pose = np.array([end_loc[0], end_loc[1], angle])

        dist = np.linalg.norm(end_loc - start_loc)
        node_separation = self.config_PT.env.node_separation
        num_nodes = int(dist / node_separation) + 1
        path = np.linspace(start_pose, end_pose, num_nodes, endpoint=True)
        
        # want path to be 3xN
        path = np.transpose(path)
        return path

    def generate_state(self, path, pose, kd_tree, MPC=None, interp_pose=None, interp_index=None):
        if interp_pose is None or interp_index is None:
            interp_pose, interp_index, closest_indices = self.localize_on_path(kd_tree, pose[:2], path, localize_to_start=self.localize_to_start)

        len_path = path.shape[1]
        state_gen_input = {'path': path, 'pose': pose, 'interp_pose': interp_pose, 'interp_index': interp_index}
        PT_state = self.state_gen(state_gen_input, self.PT_model_name, self.model_kwargs, config_object_dict=True, continuous_task=self.continuous_task)
        PT_state = np.expand_dims(PT_state, axis=0)

        return_state = {'PT_state': PT_state}
        if MPC is not None:
            MPC_actions = MPC.act(state_gen_input)
            return_state['MPC_actions'] = np.expand_dims(MPC_actions[:self.MPC_elements_in_input],0)

        return return_state


    def query_off_path(self, position, constrained_dist):
        curr_interp_pose, curr_interp_index, curr_closest_indices = self.localize_on_path(self.kd_tree, position, self.path, strict=False, localize_to_start=self.localize_to_start)
        
        xy_dev_curr = np.linalg.norm(position - curr_interp_pose[:2])
        
        if  constrained_dist < xy_dev_curr:
            return True, xy_dev_curr
        else:
            return False, xy_dev_curr
        
    def query_closest_path_node(self, x, y):
        curr_interp_pose, curr_interp_index, curr_closest_indices = self.localize_on_path(self.kd_tree, [x, y], self.path, strict=False, localize_to_start=self.localize_to_start)
        return curr_interp_pose
    
    def query_closest_path_node_plus_one(self, x, y):
        curr_interp_pose, curr_interp_index, curr_closest_indices = self.localize_on_path(self.kd_tree, [x, y], self.path, strict=False, localize_to_start=self.localize_to_start)
        desired_node_idx = int(curr_interp_index + 1)%self.path.shape[1]

        return self.path[:,desired_node_idx]