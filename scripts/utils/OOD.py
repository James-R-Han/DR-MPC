import faiss # The import order between faiss and torch matters!
import torch
 
import numpy as np
from copy import deepcopy

# Implementation of: https://arxiv.org/pdf/2204.06507.pdf
class OOD:
    def __init__(self, model, device, OOD_config):
        self.model = model
        self.device = device

        self.threshold = None
        self.index = None

        self.K = OOD_config.K 
        self.min_dataset_size = OOD_config.min_dataset_size 
        
        self.check_asserts()

        self.curr_dataset_size = 0
        self.threshold_determined = False

        self.replay_buffer_for_fitting = None
        self.data_samples_set = False

    def check_asserts(self):
        assert self.min_dataset_size >= 0, "min_dataset_size must be >= 0"
        assert self.K >= 1, "K must be >= 1"


    def determine_threshold_with_replay_buffer(self):
        if self.replay_buffer_for_fitting is None:
            return
        if self.replay_buffer_for_fitting.valid_entries < 100: 
            return
        
        indices = torch.arange(100)
        with torch.no_grad():
            obs, A, obs_prime, R, done = self.replay_buffer_for_fitting.sample_from_buffer(0, indices_to_sample=indices)
            
            self.model.run_actor(obs, deterministic=False)
            OODfeatures = self.model.base.actor.features_for_OOD
            OODfeatures = OODfeatures.cpu().numpy()
            OODfeatures_obs1 = OODfeatures / np.linalg.norm(OODfeatures, axis=1, keepdims=True)

            self.model.run_actor(obs_prime, deterministic=False)
            OODfeatures = self.model.base.actor.features_for_OOD
            OODfeatures = OODfeatures.cpu().numpy()
            OODfeatures_obs2 = OODfeatures / np.linalg.norm(OODfeatures, axis=1, keepdims=True)

        self.threshold = np.linalg.norm(OODfeatures_obs1 - OODfeatures_obs2, axis=1).mean()
        self.threshold = self.threshold / 2.0 # make more strict without having to increasing K (which increases computational complexity)

        
        self.threshold_determined = True


    def fit_model(self, replay_buffer):
        if self.threshold_determined == False:
            return
        num_entries_in_replay_buffer = replay_buffer.valid_entries        
        if num_entries_in_replay_buffer < self.K:
            return
        
        replay_buffer_for_fit = replay_buffer
        self.replay_buffer_for_fitting = replay_buffer_for_fit

        self.model.base.eval()
        OODfeatures = self.compute_OOD_features(replay_buffer_for_fit)
        # divide by L2 norm
        normalized_OOD_features = OODfeatures / torch.norm(OODfeatures, p=2, dim=1, keepdim=True) 
        
        # convert to numpy
        normalized_OOD_features = normalized_OOD_features.cpu().numpy()
        
        # FAISS implementation # GPU method doesn't work for me atm. Some package issue I think.
        self.data_dimension = normalized_OOD_features.shape[1]
        index = faiss.IndexFlatL2(self.data_dimension)
        index.add(normalized_OOD_features)
        self.index = index

        
    def ID_query(self, features=None, obs=None, model=None):
        """
        RL_input is expected to be (BxRL_input)
        
        output: 1 is ID and 0 is OOD
        """
        if self.index is None:
            return False, None


        if features is None:
            assert obs is not None and model is not None, "RL_input and model must be provided if features is None"
            if model is None:
                model = self.model

            
            with torch.no_grad():
                model.run_actor(obs, deterministic=False)
                features = model.base.actor.features_for_OOD
        
        query_feature_np = features.cpu().numpy()
        normalized_query_feature_np = query_feature_np / np.linalg.norm(query_feature_np, axis=1, keepdims=True)

        D, I = self.index.search(normalized_query_feature_np, self.K)
        distance_to_k_nearest_neighbour = D[:, -1]

        # this thing is a numpy bool so we need to convert to python bool
        distance = distance_to_k_nearest_neighbour.item() 
        ID =  distance < self.threshold
        return bool(ID), distance
        # return False


    def compute_OOD_features(self, replay_buffer):
        batch_size = 5000 # can't take in data all at once since it won't fit in GPU memory

        data_batches = replay_buffer.grab_all_data(batch_size)
        OODfeatures = []
        for obs, action, next_obs, reward, done in data_batches:
            with torch.no_grad():
                self.model.run_actor(obs, deterministic=False)
            
            # OODfeatures.append(self.model.base.shared_layers_critic.features_for_OOD)
            OODfeatures.append(self.model.base.actor.features_for_OOD)

        OODfeatures = torch.cat(OODfeatures, dim=0)

        return OODfeatures
