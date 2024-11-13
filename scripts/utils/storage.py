import torch
from copy import deepcopy
import numpy as np
import pickle

class ReplayBuffer(object):
    """ Replay Buffer"""
    def __init__(self, obs_example, config_master, buffer_size):

        self.config_master = config_master
        self.buffer_size = buffer_size
        self.device = config_master.config_general.device

        self.valid_entries = 0
        
        # This to an automated process where we have an example obs and construct these things
        # shouldn't be more than 2 recursive dictionary levels
        self.replay_buffer = {}
        self.replay_buffer['S'] = {}
        self.replay_buffer['S_prime'] = {}
        for key, value in obs_example.items():
            if isinstance(value, dict):
                self.replay_buffer['S'][key] = {}
                self.replay_buffer['S_prime'][key] = {}
                for subkey, subvalue in value.items():
                    self.replay_buffer['S'][key][subkey] = torch.zeros(buffer_size, *(subvalue.shape[1:])).type(torch.float64).to(self.device)
                    self.replay_buffer['S_prime'][key][subkey] = torch.zeros(buffer_size, *(subvalue.shape[1:])).type(torch.float64).to(self.device)
            else:
                # self.replay_buffer[key] = torch.zeros(buffer_size, *(obs_example[key].shape[1:]))
                self.replay_buffer['S'][key] = torch.zeros(buffer_size, *(value.shape[1:])).type(torch.float64).to(self.device)
                self.replay_buffer['S_prime'][key] = torch.zeros(buffer_size, *(value.shape[1:])).type(torch.float64).to(self.device)
                
        self.replay_buffer['rewards'] = torch.zeros(buffer_size, 1).type(torch.float64).to(self.device)
        self.replay_buffer['actions'] = torch.zeros(buffer_size, 2).type(torch.float64).to(self.device)
        self.replay_buffer['done'] = torch.zeros(buffer_size, 1).type(torch.float64).to(self.device)

    def full(self):
        return self.valid_entries == self.buffer_size

    def to(self, device):
        for key in self.obs:
            self.obs[key] = self.obs[key].to(device)
        for key in self.recurrent_hidden_states:
            self.recurrent_hidden_states[key] = self.recurrent_hidden_states[key].to(device)

        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, S, A, S_prime, R, done):
        
        if self.valid_entries == self.buffer_size:
            # go through and roll all tensors and insert at the end
            update_entry = -1

            for key, value in self.replay_buffer['S'].items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        self.replay_buffer['S'][key][subkey] = self.replay_buffer['S'][key][subkey].roll(-1, dims=0)
                else:
                    self.replay_buffer['S'][key] = self.replay_buffer['S'][key].roll(-1, dims=0)

            for key, value in self.replay_buffer['S_prime'].items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        self.replay_buffer['S_prime'][key][subkey] = self.replay_buffer['S_prime'][key][subkey].roll(-1, dims=0)
                else:
                    self.replay_buffer['S_prime'][key] = self.replay_buffer['S_prime'][key].roll(-1, dims=0)

            self.replay_buffer['rewards'] = self.replay_buffer['rewards'].roll(-1, dims=0)
            self.replay_buffer['actions'] = self.replay_buffer['actions'].roll(-1, dims=0)
            self.replay_buffer['done'] = self.replay_buffer['done'].roll(-1, dims=0)
            
        else:
            self.valid_entries += 1
            update_entry = self.valid_entries - 1

        for key, value in S.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    self.replay_buffer['S'][key][subkey][update_entry] = subvalue
            else:
                self.replay_buffer['S'][key][update_entry] = value

        for key, value in S_prime.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    self.replay_buffer['S_prime'][key][subkey][update_entry] = subvalue
            else:
                self.replay_buffer['S_prime'][key][update_entry] = value

        self.replay_buffer['rewards'][update_entry] = R
        self.replay_buffer['actions'][update_entry] = A
        self.replay_buffer['done'][update_entry] = done
        

    def sample_from_buffer(self, batch_size, indices_to_sample=None):
        # randomly sample from what's in the buffer
        if self.valid_entries < batch_size:
            # add a warnining
            print(f"Not enough entries in buffer to sample from. Only {self.valid_entries} entries in buffer")
            # raise ValueError("Not enough entries in buffer to sample from")
        return_S = {}
        return_S_prime = {}

        if indices_to_sample is None:
            indices_to_sample = torch.randint(0, self.valid_entries, (batch_size,))

        for key, value in self.replay_buffer['S'].items():
            if isinstance(value, dict):
                return_S[key] = {}
                for subkey, subvalue in value.items():
                    return_S[key][subkey] = subvalue[indices_to_sample]
            else:
                return_S[key] = value[indices_to_sample]

        for key, value in self.replay_buffer['S_prime'].items():
            if isinstance(value, dict):
                return_S_prime[key] = {}
                for subkey, subvalue in value.items():
                    return_S_prime[key][subkey] = subvalue[indices_to_sample]
            else:
                return_S_prime[key] = value[indices_to_sample]

        return_reward = self.replay_buffer['rewards'][indices_to_sample]
        return_action = self.replay_buffer['actions'][indices_to_sample]
        return_done = self.replay_buffer['done'][indices_to_sample]

        return return_S, return_action, return_S_prime, return_reward, return_done

    def grab_all_data(self, batch_size):
        curr_idx = 0
        return_data = []
        while curr_idx < self.valid_entries:
            end_idx = min(curr_idx + batch_size, self.valid_entries)
            indices_to_sample = torch.arange(curr_idx, end_idx)
            return_data.append(self.sample_from_buffer(0, indices_to_sample=indices_to_sample))
            curr_idx = end_idx
        
        return return_data

    def save_to_pickle(self, filename):
        numpy_buffer = self.convert_buffer_to_numpy()
        with open(filename, 'wb') as f:
            pickle.dump(numpy_buffer, f)

    def load_from_pickle(self, filename):
        with open(filename, 'rb') as f:
            buffer = pickle.load(f)

         # find number of valid entries through last non-zero element in actions
        self.valid_entries = np.where(buffer['actions'][:, 0] == 0)[0][0]
        print(f"Loaded buffer with {self.valid_entries} entries")

        self.replay_buffer = self.convert_np_buffer_to_tensor(buffer)

    def convert_np_buffer_to_tensor(self, buffer, device=None):
        if device is None:
            device = self.device

        for key, value in buffer['S'].items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    buffer['S'][key][subkey] = torch.tensor(subvalue).to(device)
            else:
                buffer['S'][key] = torch.tensor(value).to(device)

        for key, value in buffer['S_prime'].items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    buffer['S_prime'][key][subkey] = torch.tensor(subvalue).to(device)
            else:
                buffer['S_prime'][key] = torch.tensor(value).to(device)

        buffer['rewards'] = torch.tensor(buffer['rewards']).to(device)
        buffer['actions'] = torch.tensor(buffer['actions']).to(device)
        buffer['done'] = torch.tensor(buffer['done']).to(device)

        return buffer

    def convert_buffer_to_numpy(self):
        buffer_copy = deepcopy(self.replay_buffer)

        for key, value in buffer_copy['S'].items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    buffer_copy['S'][key][subkey] = subvalue.cpu().numpy()
            else:
                buffer_copy['S'][key] = value.cpu().numpy()

        for key, value in buffer_copy['S_prime'].items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    buffer_copy['S_prime'][key][subkey] = subvalue.cpu().numpy()
            else:
                buffer_copy['S_prime'][key] = value.cpu().numpy()

        buffer_copy['rewards'] = buffer_copy['rewards'].cpu().numpy()
        buffer_copy['actions'] = buffer_copy['actions'].cpu().numpy()
        buffer_copy['done'] = buffer_copy['done'].cpu().numpy()

        return buffer_copy
        