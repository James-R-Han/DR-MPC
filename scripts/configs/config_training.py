import os
import numpy as np

from .config_base import ConfigBase

class ConfigTraining:
    def __init__(self, config_general):

        # ct stands for continuous task
        online_ct = ConfigBase()
        online_ct.generate_plots_during_training = True
        online_ct.num_runs = 3

        # model training related
        online_ct.data_steps = 150 * 250
        online_ct.replay_buffer_size = online_ct.data_steps 
        online_ct.updates_per_data_step = 2
        online_ct.batch_size = 200
        online_ct.data_before_training = 100

        online_ct.save_freq = 50 * 250 # with respect to # of data steps
        self.online_ct = online_ct

        ###################################################################

        # et stands for episodic task
        online_et = ConfigBase()
        online_et.generate_plots_during_training = True
        online_et.num_runs = 3

        # model training related
        online_et.num_episodes = 500
        online_et.replay_buffer_size = online_et.num_episodes * 200 
        online_et.updates_per_data_step = 2
        online_et.batch_size = 200
        online_et.data_before_training = 200

        online_et.save_freq = 20 # with respect to # of episodes
        self.online_et = online_et
