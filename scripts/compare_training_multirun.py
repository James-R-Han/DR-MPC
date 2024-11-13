import os
import numpy as np
import matplotlib.pyplot as plt

from scripts.utils.plotting import moving_average



moving_average_smooth_factor = 2500


run_number = 0
models_to_compare_base = {'online_continuous_task': {'DR-MPC':[f'medium_vis_LB6_OODFalse_{run_number}', f'medium_vis_LB6_OODTrue_{run_number}'], 
                                                     'DRL':[f'medium_vis_LB6_OODFalse_{run_number}'], 
                                                     'ResidualDRL': [f'medium_vis_LB6_OODFalse_{run_number}' ]}}
remappings_model = {f'DR-MPC_medium_vis_LB6_OODFalse_{run_number}': 'DR-MPC (ours)',
                    f'DR-MPC_medium_vis_LB6_OODTrue_{run_number}': 'DR-MPC w/ OOD(ours)',
                    f'DRL_medium_vis_LB6_OODFalse_{run_number}': 'DRL',
                    f'ResidualDRL_medium_vis_LB6_OODFalse_{run_number}': 'ResidualDRL'}


metrics_to_compare = ['cumulative_reward_of_500.npy',
                      'num_collisions_per_500.npy',
                      'num_safety_human_raises_per_500.npy',
                      'num_safety_corridor_raises_per_500.npy',
                      'num_collisions_per_500.npy',
                      'num_corridor_hits_per_500.npy',
                      'num_incorrect_endings_per_500.npy',
                      'deviation_reward_per_500.npy',
                      'path_advancement_per_500.npy',
                      'disturbance_reward_per_500.npy']


data_folder = 'HA_and_PT_results'
results_folder = os.path.join(data_folder, f'results')
os.makedirs(results_folder, exist_ok=True)

models_to_compare = []
for training_type, model_types in models_to_compare_base.items():

    for model_type, model_indices in model_types.items():
        for model_index in model_indices:
            model_path = os.path.join(training_type, f"{model_type}_{model_index}")
            models_to_compare.append(model_path)
            print(model_path)

for metric in metrics_to_compare:
    for model in models_to_compare:
        base_model_path = os.path.join(data_folder, model)
        
        list_of_results = []
        for model_run in os.listdir(base_model_path):
            run_num = model_run.split('_')[-1]
            

            print(run_num)
            model_dir = os.path.join(base_model_path, model_run)
            metric_file = os.path.join(model_dir, 'metrics' ,metric )

            arr = np.load(metric_file)
            arr = moving_average(arr, moving_average_smooth_factor)
            list_of_results.append(arr)

        mean_arr = np.mean(list_of_results, axis=0)
        std_arr = np.std(list_of_results, axis=0)*(2/3) # if you have a low number of runs, the variance can be pretty large (due to the nature of DRL)

        x_axis = np.arange(arr.shape[0])

        name_for_model = model.split("/")[1]
        if name_for_model in remappings_model:
            name_for_model = remappings_model[name_for_model]
        plt.plot(x_axis, mean_arr, label=name_for_model)
        plt.fill_between(x_axis, mean_arr - std_arr, mean_arr + std_arr, alpha=0.3)

    # save plot
    plt.title(f"{metric} During Training")
    plt.legend(fontsize=10)
    plt.xlabel('Environment Steps')
    plt.ylabel(metric)
    plt.savefig(os.path.join(results_folder, metric + '.png'))
    plt.clf()
