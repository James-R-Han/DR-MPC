import os
import shutil
from tqdm import tqdm
from copy import deepcopy

from scripts.utils.OOD import OOD # note must be first import - there is some weird interaction between faiss and torch
from scripts.configs import ConfigMaster
from environment.HA_and_PT.human_avoidance_and_path_tracking_env import HAAndPTEnv
from scripts.models.model import Policy

from scripts.utils.storage import ReplayBuffer


from scripts.RL.sac import SACAgent
from scripts.utils.plotting import moving_average

from scripts.utils.misc import convert_to_tensor
from environment.human_avoidance.utils.action import ActionRot

import torch
import numpy as np
import matplotlib.pyplot as plt


class OnlineETTraining:

    def __init__(self):
        self.config_master = ConfigMaster()
        assert self.config_master.config_general.env.continuous_task is False, "Online ET training is only for episodic tasks"
        self.device = self.config_master.config_general.device
        self.config_training = self.config_master.config_training.online_et

        raise NotImplementedError

        
    def pure_online_et_training(self, run):
        raise NotImplementedError

if __name__ == '__main__':
    OnlineETTraining()