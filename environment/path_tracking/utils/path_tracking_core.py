import os
import scipy

import numpy as np



class PTEnvCore:
    def __init__(self):
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


    @staticmethod
    def build_kdtree(path):
        # build only on x,y positions
        path_nxdim = np.transpose(path)
        kdtree = scipy.spatial.KDTree(path_nxdim[:, :2])
        return kdtree

    @staticmethod
    def localize_on_path(kdtree, query_xy, path, strict=False, localize_to_start=False):
        # localize_to_start matters on closed loops where the robot may be at the start and then goes behind the start node --> don't want to localize to the end
        dists, indices = kdtree.query(query_xy, k=2)
        len_path = path.shape[1]

        if localize_to_start:
            # if any of the localization points are at the end of the path, then return the end of the path
            if indices[0] >  5*len_path//8 or indices[1] > 5*len_path//8:
                return path[:,0], 0, np.array([0,1])
        else:
            # if any of the localization points are at the start of the path, then return the start of the path
            if indices[0] < len_path//6 or indices[1] < len_path//6:
                return path[:,-1], len_path-1, np.array([len_path-1,0])

        between_start_and_end = abs(indices[0] - indices[1]) == len_path - 1
        # check case if there's no precise localization - there's multiple right answers for what the localization index. Ex. imagine robot at the center of a circle - any index is just as valid
        if abs(indices[0] - indices[1]) != 1 and not between_start_and_end:
            if strict is False:
                # return closest node
                interp_pose = path[:, indices[0]]
                return interp_pose, indices[0], indices
            else:
                return None, None, indices

        # check edge cases where clsoest node is at the start or end of the path
        if indices[0] == 0:
            vec1= query_xy - path[:2,0]
            vec2 = path[:2,1] - path[:2,0]
            # if angle obtuse, then point is before path
            if np.dot(vec1, vec2) < 0:
                return path[:,0], 0, indices
        elif indices[0] == path.shape[1] - 1:
            vec1 = query_xy - path[:2,-1]
            vec2 = path[:2,-2] - path[:2,-1]
            # if angle obtuse, then point is after path
            if np.dot(vec1, vec2) < 0:
                return path[:,-1], path.shape[1] - 1, indices

        point1 = path[:, indices[0]]
        point2 = path[:, indices[1]]

        # find interpolated pose
        vec_base = point2[:2] - point1[:2]
        vec_diff = query_xy[:2] - point1[:2]
        proj = np.dot(vec_diff, vec_base) / np.dot(vec_base, vec_base)

        if proj < 0:
            # need to consider index on other side of indices[0] (relative to indices[1])
            next_index = (indices[0] - (indices[1] - indices[0])) % len_path

            # project query_xy onto path[:2, next_index] and path[:2, indices[0]]
            vec_base = path[:2, next_index] - path[:2, indices[0]]
            proj = np.dot(vec_diff, vec_base) / np.dot(vec_base, vec_base)
        
            inter_index = indices[0] + proj * (next_index - indices[0])
            interp_pose = path[:, indices[0]] + proj * (path[:, next_index] - path[:, indices[0]])
        else:
            inter_index = indices[0] + proj * (indices[1] - indices[0])
            interp_pose = point1 + proj * (point2 - point1)

        return interp_pose, inter_index, indices

    @staticmethod
    def state_gen(state_gen_input, model_name, model_kwargs, config_object_dict=False, continuous_task=False):
        # feel free to add other path representations for state gen here!
        if model_name == 'MLP_local':
            state = PTEnvCore.state_gen_MLP_local_format(state_gen_input, model_kwargs, config_object_dict, continuous_task=continuous_task)
        else:
            raise ValueError(f'Invalid state gen name: {model_name}')

        return state

    @staticmethod
    def state_gen_MLP_local_format(state_gen_input, model_kwargs, config_object_dict=False, continuous_task=False):
        # trying new state representation to avoid angle wrapping issue
        # unpack inputs
        path = state_gen_input['path']
        curr_pose = state_gen_input['pose']
        interp_index = state_gen_input['interp_index']
        interp_pose = state_gen_input['interp_pose']
        if config_object_dict is False:
            lookahead = model_kwargs['lookahead']
            lookbehind = model_kwargs['lookbehind']
            frame = model_kwargs['frame']
        else:
            lookahead = model_kwargs.lookahead
            lookbehind = model_kwargs.lookbehind
            frame = model_kwargs.frame
    

        # rotate local path into current frame
        if frame == 'original' or frame == 'robot':
            curr_th = curr_pose[2]
            frame_pos = np.expand_dims(curr_pose[:2], axis=1)
        elif frame == 'path':
            curr_th = interp_pose[2]
            frame_pos = np.expand_dims(interp_pose[:2], axis=1)
        
        cos_th = np.cos(curr_th)
        sin_th = np.sin(curr_th)
        C_r_g = np.array([[cos_th, sin_th], [-sin_th, cos_th]])
        # state_xy = path[:2, :] - np.expand_dims(curr_pose[:2], axis=1)
        state_xy = path[:2, :] - frame_pos
        state_xy = np.matmul(C_r_g, state_xy)
        
        state_th = path[2, :] - curr_th
        path = np.concatenate((state_xy, np.expand_dims(state_th, axis=0)), axis=0)

        if continuous_task:
            path_aug = np.zeros((4 , path.shape[1]))
            path_aug[:2, :] = path[:2, :]
            path_aug[2, :] = np.cos(path[2, :])
            path_aug[3, :] = np.sin(path[2, :])

            len_path = path.shape[1]
            immediate_next_pose_on_path = int(np.ceil(interp_index)) 
            start_idx = immediate_next_pose_on_path - lookbehind
            end_idx = immediate_next_pose_on_path + lookahead -1
            # get all the path idx by wrapping around (%len_path_max_idx)
            path_idxes = np.arange(start_idx, end_idx+1) % len_path
            state_global = path_aug[:, path_idxes]

            # make a 4N x 1 state
            state = state_global.flatten()
            return state

        else:

            # augment path: change theta to dx, dy and add an index indicating where it is along the path
        
            path_aug = np.zeros((4 , path.shape[1]))
            path_aug[:2, :] = path[:2, :]

            path_aug[2, :] = np.cos(path[2, :])
            path_aug[3, :] = np.sin(path[2, :])


            immediate_next_pose_on_path = int(np.ceil(interp_index))
            max_idx = path_aug.shape[1] - 1
            num_unique_poses_ahead = min(lookahead, max_idx - immediate_next_pose_on_path + 1)
            state_global = path_aug[:, immediate_next_pose_on_path:immediate_next_pose_on_path+num_unique_poses_ahead]
        
            if num_unique_poses_ahead < lookahead:
                goal_pose = path_aug[:, -1]
                # concat state with last pose repeated
                state_global = np.concatenate((state_global, np.tile(goal_pose, (lookahead-num_unique_poses_ahead, 1)).T), axis=1)
        
            min_idx = 0
            num_unique_poses_behind = min(lookbehind, immediate_next_pose_on_path - min_idx)
            state_global = np.concatenate((path_aug[:, immediate_next_pose_on_path-num_unique_poses_behind:immediate_next_pose_on_path], state_global), axis=1)
        
            if num_unique_poses_behind < lookbehind:
                start_pose = path_aug[:, 0]
                # concat state with first pose repeated
                state_global = np.concatenate((np.tile(start_pose, (lookbehind-num_unique_poses_behind, 1)).T, state_global), axis=1)
        
            state = state_global.flatten()
            return state
