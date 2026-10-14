import numpy as np
import torch

class ActionGeneration:
    def __init__(self, strategy_name, max_v, max_w, model=None, model_name=None, exploit=True):
        self.model = model
        self.model_name = model_name
        self.strategy_name = strategy_name
        self.strategy = self.select(strategy_name)
        self.max_v = max_v
        self.max_w = max_w
        self.exploit = exploit



    def select(self, strategy_name):
        strategies = ["random", "model", "pp", "mpc"]
        if strategy_name not in strategies:
            raise ValueError("Strategy name not found.")

        if strategy_name == "model" or strategy_name == "mpc":
            return self.model

        return getattr(self, strategy_name)

    def generate_action(self, state, state_gen_input, strategy_name_override=None):
        infos = {}
        strategy_name = strategy_name_override if strategy_name_override is not None else self.strategy_name
        if strategy_name == "model":
            if self.model_name == 'DRL_MLP' or self.model_name == 'DRL_MLP_local':
                tensor_state = torch.from_numpy(state).float().to(self.model.device)
            elif self.model_name == 'DRL_conv':
                # add channels and batch size
                tensor_state = torch.Tensor(state).unsqueeze(0).unsqueeze(0).to(self.model.device)
            # reordered_tensor_state = tensor_state.permute(2, 0, 1).unsqueeze(0)
            # base_dist = self.model.actor(reordered_tensor_state)
                
            base_dist, infos = self.model.actor(tensor_state)
            if self.exploit:
                action = base_dist.mean
            else:
                action = base_dist.sample()
            
            # action = base_dist.sample()
            base_action = action.squeeze(0).cpu().detach().numpy()
        elif strategy_name == "mpc":
            base_action = self.model.act(state, state_gen_input)
        else:
            base_action = self.strategy(state, state_gen_input)
        clamped_action = self.clamp(base_action)
        # return base_action
        return clamped_action, infos

    def random(self, state, state_gen_input):
        v = np.random.uniform(0, self.max_v)
        w = np.random.uniform(-self.max_w, self.max_w)
        return np.array([v, w])

    def pp(self, state, state_gen_input):
        curr_pose = state_gen_input['pose']
        path = state_gen_input['path']
        len_path = path.shape[1]
        interp_index = state_gen_input['interp_index']
        chase_index = int(min( np.ceil(interp_index), len_path - 1))
        chase_pose = path[:, chase_index]

        while np.linalg.norm(curr_pose[:2] - chase_pose[:2]) < 0.3 and chase_index != len_path - 1:
            chase_index = chase_index + 1
            chase_pose = path[:2, chase_index]

        # compute chase_pose relative to curr_pose
        curr_th = curr_pose[2]
        cos_th = np.cos(curr_th)
        sin_th = np.sin(curr_th)
        C_r_g = np.array([[cos_th, sin_th], [-sin_th, cos_th]])
        chase_xy_intermediate = chase_pose[:2] - curr_pose[:2]
        path_rel_xy = np.matmul(C_r_g, chase_xy_intermediate)
        y = path_rel_xy[1]
        x = path_rel_xy[0]
        w_div_v = 2*y/(x**2 + y**2)

        if w_div_v < 1:
            v = 1
            w = w_div_v
        else:
            w = np.sign(w_div_v)
            v = 1/np.abs(w_div_v)

        return np.array([v, w])

    def clamp(self, action):
        v = np.clip(action[0], 0, self.max_v)
        w = np.clip(action[1], -self.max_w, self.max_w)
        return np.array([v, w])

    # state_gen_input = {'path': path, 'pose': curr_pose, 'interp_pose': interp_pose, 'interp_index': interp_index,
    #                    'global_map_occ': global_map_occ, 'global_map_params': global_map_params,
    #                    'closest_indicies': closest_indices}



