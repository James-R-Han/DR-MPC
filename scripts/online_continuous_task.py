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


class OnlineCTTraining:

    def __init__(self):
        self.config_master = ConfigMaster()
        self.device = self.config_master.config_general.device
        self.config_training = self.config_master.config_training.online_ct

        self.policy = self.config_master.config_general.model.policy
        self.create_base_folder()
        self.generate_plots_during_training = self.config_training.generate_plots_during_training

        num_runs = self.config_training.num_runs
        for run in range(num_runs):
            self.env = HAAndPTEnv(self.config_master, seed_addition=run)
            example_S = self.env.reset()
            self.replay_buffer = ReplayBuffer(example_S, self.config_master, self.config_training.replay_buffer_size)

            self.actor_critic = Policy(self.config_master.config_general.model.policy, self.env.config_master)
            self.SAC = SACAgent(self.config_master, self.config_master.config_SAC,  self.actor_critic)
            self.curr_epoch = 0

            # OOD Related
            # create OOD model
            self.use_OOD = self.config_master.config_general.OOD.use_OOD
            if self.use_OOD:
                self.OOD_model = OOD(self.actor_critic, self.device, self.config_master.config_general.OOD)
            
                self.safety_module = self.config_master.config_general.safety_module.name
                self.safety_module_params = self.config_master.config_general.safety_module.params
        
            self.bool_track_beta = self.config_master.config_general.model.has_beta and  self.config_master.config_general.model.use_beta_adjustment

            self.create_folders(run)
            result_of_run = self.pure_online_ct_training(run)
            if result_of_run == "kill_run":
                print("Run couldn't complete")
                # delete this run folder
                # shutil.rmtree(self.base_save_folder_run)

    def create_base_folder(self):
        save_folder = os.path.join(self.config_master.config_general.save_folder, 'online_continuous_task')

        os.makedirs(save_folder, exist_ok=True)        

        identifier = self.config_master.identifier
        model_num = 0
        while os.path.exists(os.path.join(save_folder,f'{identifier}_{model_num}')):
            model_num += 1
        name_lastpart = f'{identifier}_{model_num}'
        self.base_save_folder = os.path.join(save_folder, name_lastpart)

    def create_folders(self, run):
        base_save_folder_run = os.path.join(self.base_save_folder, f'run_{run}')
        self.base_save_folder_run = base_save_folder_run

        self.plots_dir = os.path.join(base_save_folder_run, 'plots')
        self.trained_models_dir = os.path.join(base_save_folder_run, 'trained_models')
        self.metrics_dir = os.path.join(base_save_folder_run, 'metrics')
        
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.trained_models_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        shutil.copytree('scripts/configs', os.path.join(base_save_folder_run, 'configs'))

        self.plots_dir_videos = os.path.join(self.plots_dir, 'videos')
        os.makedirs(self.plots_dir_videos, exist_ok=True)

        
    def pure_online_ct_training(self, run):
        ### SETUP ###
        data_steps = self.config_training.data_steps
        updates_per_data = self.config_training.updates_per_data_step
        data_before_training = self.config_training.data_before_training

        save_frequency = self.config_training.save_freq

        track_critic_loss, track_actor_loss = [], []
        # this alpha is the entropy of SAC - not the fusion param in DR-MPC
        if self.SAC.entropy_version == 'all_things':
            track_alphas_entropy_param = {i: [] for i in range(self.SAC.num_alphas)}
        else:
            track_alpha_entropy_param = []
        track_entropies = []

        track_critic_avg_Q, track_critic_max_Q, track_critic_min_Q = [], [], []
        cumulative_reward_of_500 = np.zeros(500)
        track_cumulative_reward_of_500 = []
        track_beta_adjustments = []
        
        num_collisions_per_500, num_safety_human_raises_per_500, num_corridor_hits_per_500, num_safety_corridor_raises_per_500 = np.zeros(500), np.zeros(500), np.zeros(500), np.zeros(500)
        track_num_collisions_per_500, track_num_safety_human_raises_per_500, track_num_corridor_hits_per_500, track_num_safety_corridor_raises_per_500 = [], [], [], []
        deviation_reward_per_500, track_deviation_reward_per_500 = np.zeros(500), []
        path_advancement_per_500, track_path_advancement_per_500 = np.zeros(500), []
        disturbance_reward_per_500, track_disturbance_reward_per_500 = np.zeros(500), []
        num_incorrect_endings_per_500, track_num_incorrect_endings_per_500 = np.zeros(500), []
        num_actuation_terminations_per_500, track_num_actuation_terminations_per_500 = np.zeros(500), []

        if self.use_OOD:
            num_ID_per_500, track_avg_ID_per_500 = np.zeros(500), []

        if self.policy == 'ResidualDRL':
            v_adjustments_of_500, track_v_adjustments_of_500 = np.zeros(500), [] # tracking magnitude
            w_adjustments_of_500, track_w_adjustments_of_500 = np.zeros(500), [] # tracking magnitude

        # Start of training process
        training_step = 0
        S = self.env.reset()

        
        ### LOOP START ###
        S = convert_to_tensor(S, self.device)
        done = False
        vid_steps = -1
        for data_step in tqdm(range(data_steps)):
            ## STEP IN ENVIRONENT ##
            if done:
                done = False
                S = self.env.soft_reset(full_done, info, S)

                # if soft reset fails (pretty rare), we don't count this run
                if S is None:
                    return "kill_run"
                S = convert_to_tensor(S, self.device)

            if self.generate_plots_during_training: 
                if data_step % save_frequency == 0 or data_step == data_steps-501:
                    if vid_steps == -1:
                        self.env.create_video_continuous_task(f'{self.plots_dir_videos}/vid_{data_step}.mp4', S['PT'])
                        vid_steps = 0

            self.SAC.model.eval()
            self.SAC.model_target.eval()
            with torch.no_grad():
               action_model, action_execute, action_log_probs, model_info = self.actor_critic.run_actor(S, deterministic=False)
                        
            if self.use_OOD:
                res_ID_query, distance_of_OOD = self.OOD_model.ID_query(features=self.actor_critic.base.actor.features_for_OOD)
                model_info['ID'] = res_ID_query

                if res_ID_query is False: 
                    action_execute_np = torch.squeeze(action_execute, dim=0).cpu().numpy()
                    
                    if self.safety_module == 'cvmm_diverse':
                        action_for_sim, info = self.env.cvmm_diverse_safety_pipeline(action_execute_np, self.safety_module_params['lookahead_steps'], self.safety_module_params['action_set'])
                    else:
                        raise NotImplementedError

                    model_info['RL_action_bool'] = info['RL_action_bool']
                else:
                    model_info['RL_action_bool'] = 'NA'
                    action_for_sim = ActionRot(action_execute[0, 0].item(), action_execute[0, 1].item())
            
                # here action_execute may not equal action_model since it depends on OOD and safe policy
                # Note this OOD module would have to be adjusted for residual RL. We would have to adjust action_model = action_for_sim - MPC_action (ie. solve for what the action_model would have been)
                action_model[0,0] = action_for_sim.v
                action_model[0,1] = action_for_sim.w
            else:
                action_for_sim = ActionRot(action_execute[0, 0].item(), action_execute[0, 1].item())
            
            S_prime, full_reward, full_done, info, done_info = self.env.step(action_for_sim, model_info=model_info)
            reward, done = full_reward['R'], full_done['done']
            S_prime = convert_to_tensor(S_prime, self.device)
            
            self.replay_buffer.insert(S, action_model, S_prime, reward, full_done['done_for_replay'])

            ## TRACKING ##
            # termination conditions
            cumulative_reward_of_500 = np.roll(cumulative_reward_of_500, 1)
            cumulative_reward_of_500[0] = reward
            num_collisions_per_500 = np.roll(num_collisions_per_500, 1)
            num_collisions_per_500[0] = done_info['collision']
            num_safety_human_raises_per_500 = np.roll(num_safety_human_raises_per_500, 1)
            num_safety_human_raises_per_500[0] = done_info['safety_human_raise']    
            num_corridor_hits_per_500 = np.roll(num_corridor_hits_per_500, 1)
            num_corridor_hits_per_500[0] = done_info['corridor_hit']
            num_safety_corridor_raises_per_500 = np.roll(num_safety_corridor_raises_per_500, 1)
            num_safety_corridor_raises_per_500[0] = done_info['safety_corridor_raise']           
            num_incorrect_endings_per_500 = np.roll(num_incorrect_endings_per_500, 1)
            num_incorrect_endings_per_500[0] = done_info['deviation_end_of_path']
            num_actuation_terminations_per_500 = np.roll(num_actuation_terminations_per_500, 1)
            num_actuation_terminations_per_500[0] = done_info['actuation_termination']

            # track reward shaping items
            deviation_reward_per_500 = np.roll(deviation_reward_per_500, 1)
            deviation_reward_per_500[0] = info['deviation']
            path_advancement_per_500 = np.roll(path_advancement_per_500, 1)
            path_advancement_per_500[0] = info['path_advancement']
            disturbance_reward_per_500 = np.roll(disturbance_reward_per_500, 1)
            disturbance_reward_per_500[0] = info['disturbance_v'] + info['disturbance_th']

            # store
            track_cumulative_reward_of_500.append(np.sum(cumulative_reward_of_500))
            track_num_collisions_per_500.append(np.sum(num_collisions_per_500))
            track_num_safety_human_raises_per_500.append(np.sum(num_safety_human_raises_per_500))
            track_num_corridor_hits_per_500.append(np.sum(num_corridor_hits_per_500))
            track_num_safety_corridor_raises_per_500.append(np.sum(num_safety_corridor_raises_per_500))
            track_num_incorrect_endings_per_500.append(np.sum(num_incorrect_endings_per_500))
            track_deviation_reward_per_500.append(np.sum(deviation_reward_per_500))
            track_path_advancement_per_500.append(np.sum(path_advancement_per_500))
            track_disturbance_reward_per_500.append(np.sum(disturbance_reward_per_500))

            if self.use_OOD:
                num_ID_per_500 = np.roll(num_ID_per_500, 1)
                num_ID_per_500[0] = res_ID_query
                track_avg_ID_per_500.append(np.mean(num_ID_per_500) * 100)

            if self.policy == 'ResidualDRL':
                v_adjustments_of_500 = np.roll(v_adjustments_of_500, 1)
                v_adjustments_of_500[0] = abs(model_info['v_adjustment'])
                w_adjustments_of_500 = np.roll(w_adjustments_of_500, 1)
                w_adjustments_of_500[0] = abs(model_info['w_adjustment'])
                track_v_adjustments_of_500.append(np.mean(v_adjustments_of_500))
                track_w_adjustments_of_500.append(np.mean(w_adjustments_of_500))


            if vid_steps != -1:
                vid_steps += 1

            if self.generate_plots_during_training and vid_steps == 500:
                self.env.render()
                self.env.no_video()
                vid_steps = -1

            if self.use_OOD: 
                if data_step % 50 == 0:
                    self.OOD_model.replay_buffer_for_fitting = self.replay_buffer
                    self.OOD_model.determine_threshold_with_replay_buffer()

                if data_step % 200 == 0:
                    self.OOD_model.fit_model(self.replay_buffer)

            S = S_prime

            if data_step < data_before_training:
                continue

            ## TRAINING ##
            for update in range(updates_per_data):
                S_for_update, A_for_update, S_prime_for_update, R_for_update, done_for_update = self.replay_buffer.sample_from_buffer(self.config_training.batch_size)
                self.SAC.model.train()
                self.SAC.model_target.eval()
                critic_loss, actor_loss, debug_quantities = self.SAC.update(S_for_update, A_for_update, S_prime_for_update, R_for_update, done_for_update, training_step)

                track_critic_loss.append(critic_loss)
                if self.SAC.entropy_version == 'all_things':
                    for i in range(self.SAC.num_alphas):
                        track_alphas_entropy_param[i].append(self.SAC.log_alphas[i].exp().item())
                    
                else:
                    track_alpha_entropy_param.append(self.SAC.alpha.item())
                if actor_loss is not None:
                    track_actor_loss.append(actor_loss)
                else:
                    track_actor_loss.append(track_actor_loss[-1])

                # track_critic_avg_Q.append(debug_quantities['debug_value_critic'])
                debug_value_critic = debug_quantities['debug_value_critic']
                track_critic_avg_Q.append(debug_value_critic[0])
                track_critic_max_Q.append(debug_value_critic[1])
                track_critic_min_Q.append(debug_value_critic[2])

                track_entropies.append(debug_quantities['entropies'])

                if self.bool_track_beta:
                    track_beta_adjustments.append(self.actor_critic.base.actor.beta_adjustment.item())

                training_step += 1
            
            if self.policy == 'DR-MPC' and data_step % 250 == 0:
                self.SAC.model.base.actor.dist_HA_actions.update_fixed_std()

            ## PLOTTING ##
            if data_step % 5000 == 0 or data_step == data_steps - 1:
                # plot for cumulative reward
                smooth_factor = 1500 if data_step > 1500 else data_step-1
                smooth_track_cumulative_reward_of_500 = moving_average(track_cumulative_reward_of_500, smooth_factor)
                x_axis = np.arange(0, len(smooth_track_cumulative_reward_of_500))
                plt.plot(x_axis, smooth_track_cumulative_reward_of_500)
                plt.title(f"Smoothed Cumulative Reward of 500: data step {data_step}")
                plt.xlabel("Data Step")
                plt.ylabel("Cumulative Reward")
                plt.savefig(f"{self.plots_dir}/cumulative_reward_smoothed.png")
                plt.clf()


                x_axis = np.arange(0, len(track_cumulative_reward_of_500))
                plt.plot(x_axis, track_cumulative_reward_of_500)
                plt.title(f"Cumulative Reward of 500: data step {data_step}")
                plt.xlabel("Data Step")
                plt.ylabel("Cumulative Reward")
                plt.savefig(f"{self.plots_dir}/cumulative_reward.png")
                plt.clf()


                # plot for num collisions
                x_axis = np.arange(0, len(track_num_collisions_per_500))
                plt.plot(x_axis, track_num_collisions_per_500)
                plt.title(f"Num Collisions of 500: data step {data_step}")
                plt.xlabel("Data Step")
                plt.ylabel("Num Collisions")
                plt.savefig(f"{self.plots_dir}/num_collisions.png")
                plt.clf()

                # plot for num safety raises
                smooth_track_num_safety_raises_per_500 = moving_average(track_num_safety_human_raises_per_500, smooth_factor)
                x_axis = np.arange(0, len(smooth_track_num_safety_raises_per_500))
                plt.plot(x_axis, smooth_track_num_safety_raises_per_500)
                plt.title(f"Smoothed Num Safety Raises of 500: data step {data_step}")
                plt.xlabel("Data Step")
                plt.ylabel("Num Safety Raises")
                plt.savefig(f"{self.plots_dir}/num_safety_raises_smoothed.png")
                plt.clf()

                x_axis = np.arange(0, len(track_num_safety_human_raises_per_500))
                plt.plot(x_axis, track_num_safety_human_raises_per_500)
                plt.title(f"Num Safety Raises of 500: data step {data_step}")
                plt.xlabel("Data Step")
                plt.ylabel("Num Safety Raises")
                plt.savefig(f"{self.plots_dir}/num_safety_raises.png")
                plt.clf()

                # plot for num corridor hits
                x_axis = np.arange(0, len(track_num_corridor_hits_per_500))
                plt.plot(x_axis, track_num_corridor_hits_per_500)
                plt.title(f"Num Corridor Hits of 500: data step {data_step}")
                plt.xlabel("Data Step")
                plt.ylabel("Num Corridor Hits")
                plt.savefig(f"{self.plots_dir}/num_corridor_hits.png")
                plt.clf()

                # plot for num corridor safety raises
                x_axis = np.arange(0, len(track_num_safety_corridor_raises_per_500))
                plt.plot(x_axis, track_num_safety_corridor_raises_per_500)
                plt.title(f"Num Corridor Safety Raises of 500: data step {data_step}")
                plt.xlabel("Data Step")
                plt.ylabel("Num Corridor Safety Raises")
                plt.savefig(f"{self.plots_dir}/num_corridor_safety_raises.png")
                plt.clf()

                # plot for num incorrect endings
                x_axis = np.arange(0, len(track_num_incorrect_endings_per_500))
                plt.plot(x_axis, track_num_incorrect_endings_per_500)
                plt.title(f"Num Incorrect Endings of 500: data step {data_step}")
                plt.xlabel("Data Step")
                plt.ylabel("Num Incorrect Endings")
                plt.savefig(f"{self.plots_dir}/num_incorrect_endings.png")
                plt.clf()

                if self.use_OOD:
                    # plot for percent ID
                    x_axis = np.arange(0, len(track_avg_ID_per_500))
                    plt.plot(x_axis, track_avg_ID_per_500)
                    plt.title(f"Average ID % over 500: data step {data_step}")
                    plt.xlabel("Data Step")
                    plt.ylabel("Percent ID")
                    plt.savefig(f"{self.plots_dir}/percent_ID.png")
                    plt.clf()

                
                if self.policy == 'ResidualDRL':
                    x_axis_v = np.arange(0, len(track_v_adjustments_of_500))
                    x_axis_r = np.arange(0, len(track_w_adjustments_of_500))
                    plt.plot(x_axis_v, track_v_adjustments_of_500, label='v')
                    plt.plot(x_axis_r, track_w_adjustments_of_500, label='r')
                    plt.title(f"Action Adjustments of 500: data step {data_step}")
                    plt.xlabel("Data Step")
                    plt.ylabel("Average Action Adjustment")
                    plt.legend()
                    plt.savefig(f"{self.plots_dir}/action_adjustments.png")
                    plt.clf()

                # model related plotting
                self.SAC.save(self.trained_models_dir, data_step)

                # plot critic loss
                smooth_track_critic_loss = moving_average(track_critic_loss, 20)
                x_axis = np.arange(0, len(smooth_track_critic_loss))
                plt.plot(x_axis, smooth_track_critic_loss, label='train')
                # plt.plot(x_axis_val, track_val_critic_loss, label='val')
                plt.legend()
                plt.title(f"Critic Loss")
                plt.xlabel("Training Step")
                plt.ylabel("Critic Loss")
                plt.savefig(f"{self.plots_dir}/critic_loss.png")
                plt.clf()

                # plot actor_loss
                smooth_track_actor_loss = moving_average(track_actor_loss, 20)
                x_axis = np.arange(0, len(smooth_track_actor_loss))
                plt.plot(x_axis, smooth_track_actor_loss, label='train')
                # plt.plot(x_axis_val, track_val_actor_loss, label='val')
                plt.legend()
                plt.title(f"Actor Loss")
                plt.xlabel("Training Step")
                plt.ylabel("Actor Loss")
                plt.savefig(f"{self.plots_dir}/actor_loss.png")
                plt.clf()

                # plot critic Q values
                x_axis = np.arange(0, len(track_critic_avg_Q))
                plt.plot(x_axis, track_critic_avg_Q, label="avg")
                plt.plot(x_axis, track_critic_max_Q, label="max")
                plt.plot(x_axis, track_critic_min_Q, label="min")
                plt.legend()
                plt.title(f"Critic Q values")
                plt.xlabel("Training Step")
                plt.ylabel("Q value")
                plt.savefig(f"{self.plots_dir}/critic_Q.png")
                plt.clf()

                try:
                    # plot entropies
                    entropies_np = np.vstack(track_entropies)
                    x_axis = np.arange(0, entropies_np.shape[0])
                    plt.plot(x_axis, entropies_np[:,0], label=f"entropy_discrete")
                    plt.legend()
                    plt.title(f"Entropies Discrete")
                    plt.xlabel("Training Step")
                    plt.ylabel("Entropy Discrete")
                    plt.savefig(f"{self.plots_dir}/entropies_discrete.png")
                    plt.clf()

                    if len(entropies_np.shape[1] > 1):
                        for i in range(1, entropies_np.shape[1]):
                            raw_entropies = entropies_np[:,i]
                            smoothed_entropies = moving_average(raw_entropies, 20)
                            x_axis = np.arange(0, len(smoothed_entropies))
                            plt.plot(x_axis, smoothed_entropies, label=f"entropy_{i}")
                        plt.legend()
                        plt.title(f"Entropies General")
                        plt.xlabel("Training Step")
                        plt.ylabel("Smoothed Entropy")
                        plt.savefig(f"{self.plots_dir}/entropies_general.png")
                        plt.clf()

                except:
                    pass


                if self.SAC.entropy_version == 'all_things':
                    # plot alpha
                    x_axis = np.arange(0, len(track_alphas_entropy_param[0]))
                    for i in range(self.SAC.num_alphas):
                        plt.plot(x_axis, track_alphas_entropy_param[i], label=f"alpha_{i}")
                    plt.title(f"Alpha Params in SAC")
                    plt.xlabel("Training Step")
                    plt.ylabel("Value of Alpha")
                    plt.legend()
                    plt.savefig(f"{self.plots_dir}/alpha_entropy_param.png")
                    plt.clf()
                else:
                    x_axis = np.arange(0, len(track_alpha_entropy_param))
                    plt.plot(x_axis, track_alpha_entropy_param)
                    plt.title(f"Alpha Param in SAC")
                    plt.xlabel("Training Step")
                    plt.ylabel("Value of Alpha")
                    plt.savefig(f"{self.plots_dir}/alpha_entropy_param.png")
                    plt.clf()

                

                if self.bool_track_beta:
                    # plot beta
                    x_axis = np.arange(0, len(track_beta_adjustments))
                    plt.plot(x_axis, track_beta_adjustments)
                    plt.title(f"Beta Adjustments")
                    plt.xlabel("Training Step")
                    plt.ylabel("Beta Adjustment")
                    plt.savefig(f"{self.plots_dir}/beta_adjustments.png")
                    plt.clf()

            # save the rewards, safety raises, etc. as numpy array in metrics dir
            np.save(f"{self.metrics_dir}/cumulative_reward_of_500.npy", track_cumulative_reward_of_500)

            np.save(f"{self.metrics_dir}/num_collisions_per_500.npy", track_num_collisions_per_500)
            np.save(f"{self.metrics_dir}/num_safety_human_raises_per_500.npy", track_num_safety_human_raises_per_500)
            np.save(f"{self.metrics_dir}/num_corridor_hits_per_500.npy", track_num_corridor_hits_per_500)
            np.save(f"{self.metrics_dir}/num_safety_corridor_raises_per_500.npy", track_num_safety_corridor_raises_per_500)
            np.save(f"{self.metrics_dir}/num_actuation_terminations_per_500.npy", track_num_actuation_terminations_per_500)

            np.save(f"{self.metrics_dir}/num_incorrect_endings_per_500.npy", track_num_incorrect_endings_per_500)
            np.save(f"{self.metrics_dir}/deviation_reward_per_500.npy", track_deviation_reward_per_500)
            np.save(f"{self.metrics_dir}/path_advancement_per_500.npy", track_path_advancement_per_500)
            np.save(f"{self.metrics_dir}/disturbance_reward_per_500.npy", track_disturbance_reward_per_500)
            
            if self.use_OOD:
                np.save(f"{self.metrics_dir}/avg_ID_per_500.npy", track_avg_ID_per_500)
   
        return "run_complete"

if __name__ == '__main__':
    OnlineCTTraining()