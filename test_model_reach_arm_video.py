import minari
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from cnmp_ import CNMP_H, CNMP_H_Small, generate_trajectory # Assuming these are your custom modules
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from utils import sample_from_hull, calculate_average_jerk_concise
from copy import deepcopy
from gymnasium.wrappers import RecordVideo

video_folder = "reach_arm_tests_8"

# --- Model and Data Loading (Unchanged) ---
model = CNMP_H_Small(d_x=1, d_y=8, d_SM=8).double()
model.load_state_dict(torch.load("save/best_models_reach_arm_8/model.pth"))
action_data = np.load('data/reach_arm_actions_8.npy')  # shape (25, 451, 30)
observation_data = np.load('data/reach_arm_observations_8.npy')  # shape (25, 451, 42)
observation_data_all = np.load('data/reach_arm_observations_all.npy')  # shape (25, 451, 42)

#print(action_data.shape, observation_data.shape, observation_data_all.shape)

skip_indices = [13] #full
#skip_indices = [0, 1, 2, 4, 6, 7, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24] # v1
#skip_indices = [0, 1, 2, 4, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24] # v2
#skip_indices = [0, 1, 2, 4, 6, 7, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21, 22, 24] # v3
#skip_indices = [0, 1, 2, 4, 6, 7, 8, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21, 22, 24] # v4
#skip_indices = [0, 1, 2, 4, 6, 7, 8, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 24] # v5

model.eval()

dataset = minari.load_dataset('D4RL/relocate/human-v2')
normalization_values = {"actions_min": [], 
                        "actions_max": [],
                        "context_min": [],
                        "context_max": [], 
                        "context_all_min": [],
                        "context_all_max": []}

time_len = 201

GRASP_THRESHOLD = 0.076

num_data = action_data.shape[0]

Y = np.zeros((num_data, time_len, 8))
Y[:, 1:] = action_data[:num_data]
C = np.zeros((num_data, 6))
for i in range(num_data):
    C[i, :3] = observation_data[i, 0, 30:33]
    C[i, 3:] = observation_data[i, 0, 42:45]  # add the first observation as context

# normalize Y and C by dimensions
for dim in range(Y.shape[-1]):
    Y_min = np.min(Y[:, :, dim], axis=(0, 1), keepdims=True)
    Y_max = np.max(Y[:, :, dim], axis=(0, 1), keepdims=True)

    normalization_values['actions_min'].append(Y_min)
    normalization_values['actions_max'].append(Y_max)

for dim in range(C.shape[-1]):
    C_min = np.min(C[:, dim], axis=0, keepdims=True)
    C_max = np.max(C[:, dim], axis=0, keepdims=True)

    normalization_values['context_min'].append(C_min)
    normalization_values['context_max'].append(C_max)


C_all = np.zeros((observation_data_all.shape[0], 6))
for i in range(observation_data_all.shape[0]):
    C_all[i, :3] = observation_data_all[i, 0, 30:33]
    C_all[i, 3:] = observation_data_all[i, 0, 42:45]  # add the first observation as context
for dim in range(C_all.shape[-1]):
    C_min_all = np.min(C_all[:, dim], axis=0, keepdims=True)
    C_max_all = np.max(C_all[:, dim], axis=0, keepdims=True)
    normalization_values['context_all_min'].append(C_min_all)
    normalization_values['context_all_max'].append(C_max_all)

normalized_training_contexts_all = C_all.copy()
for dim in range(C_all.shape[-1]):
    normalized_training_contexts_all[:, dim] = (C_all[:, dim] - normalization_values['context_all_min'][dim][0]) / \
                         (normalization_values['context_all_max'][dim][0] - normalization_values['context_all_min'][dim][0] + 1e-8)
    
normalized_training_contexts = C.copy()
for dim in range(C.shape[-1]):
    normalized_training_contexts[:, dim] = (C[:, dim] - normalization_values['context_min'][dim][0]) / \
                         (normalization_values['context_max'][dim][0] - normalization_values['context_min'][dim][0] + 1e-8)

convex_hull = ConvexHull(C[:, [3, 4]].copy())
num_samples = 1000

np_seed = 42
random_samples = sample_from_hull(convex_hull, num_samples, seed=np_seed)
#random_samples = C[:, [9, 10]].copy() 

test_contexts = []
# --- Corrected Evaluation Loop with Correct Joint Names and API ---
std_list = []
error_list = []
jerk_list = []

for i in range(num_samples):

    if i < 100:
        continue

    temp_name_prefix = f"reach_arm_training_{i}"

    env = dataset.recover_environment(max_episode_steps=500, render_mode='rgb_array')

    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix=temp_name_prefix,
        episode_trigger=lambda x: True # Record this one episode
    )

    init_obs, _ = env.reset(seed=420+i)
    context_from_hull = random_samples[i]

    full_init_state = env.unwrapped.get_env_state()

    new_state_dict = {
        'qpos': full_init_state['qpos'].copy(),
        'qvel': full_init_state['qvel'].copy(),
        'target_pos': full_init_state['target_pos'].copy(),
        'obj_pos': np.array([context_from_hull[0], context_from_hull[1], 0.035]),
    }
    env.unwrapped.set_env_state(new_state_dict)

    cnmp_normalized = Y[0, 0, :].copy()
    for dim in range(cnmp_normalized.shape[-1]):
        cnmp_normalized[dim] = (cnmp_normalized[dim] - normalization_values['actions_min'][dim][0,0]) / \
                               (normalization_values['actions_max'][dim][0,0] - normalization_values['actions_min'][dim][0,0] + 1e-8)

    # --- Trajectory Generation and Execution (Unchanged) ---
    cnmp_obs = np.array([np.concatenate((np.array([0.0]), cnmp_normalized), axis=-1)])

    context = np.zeros((6))  # Adjusted context size to match the model's expectation
    init_state_dict = env.unwrapped.get_env_state()
    context[0:3] = env.unwrapped._get_obs()[30:33].copy()
    context[3:6] = init_state_dict['obj_pos'].copy()

    temp_context = deepcopy(context)

    for dim in range(6):
       temp_context[dim] = (temp_context[dim] - normalization_values['context_all_min'][dim][0]) / \
                            (normalization_values['context_all_max'][dim][0] - normalization_values['context_all_min'][dim][0] + 1e-8)

    test_contexts.append(temp_context)

    for dim in range(6):
        context[dim] = (context[dim] - normalization_values['context_min'][dim][0]) / \
                       (normalization_values['context_max'][dim][0] - normalization_values['context_min'][dim][0] + 1e-8)
        #print("%.3f" % context[dim].item(), end=' ')

    action_trajectory, action_std_trajectory = generate_trajectory(model, cnmp_obs, context, return_std=True)
    action_trajectory = action_trajectory.squeeze(0).detach().numpy()
    action_std_trajectory = action_std_trajectory.squeeze(0).detach().numpy()
    std_list.append(action_std_trajectory.mean())

    avg_jerk = calculate_average_jerk_concise(action_trajectory)
    jerk_list.append(avg_jerk)
    #print(action_std_trajectory.mean())


    for dim in range(action_trajectory.shape[-1]):
        action_trajectory[:, dim] = (action_trajectory[:, dim] * 
                                     (normalization_values['actions_max'][dim] - normalization_values['actions_min'][dim]) + 
                                     normalization_values['actions_min'][dim])

    init_abs_obj = init_state_dict['obj_pos'].copy()

    action = np.ones((30,)) * -1.0
    mcp_joint_indices = [8, 12, 16, 21]
    for idx in mcp_joint_indices:
        action[idx] = 0.0
    action[[27, 28]] = 0
    action[29] = 1.0
    
    for step_num in range(450):

        action[:8] = action_trajectory[step_num]
        obs, reward, terminated, truncated, info = env.step(action)

        state_dict = env.unwrapped.get_env_state()

        absolute_palm_pos = env.unwrapped.data.site_xpos[env.unwrapped.S_grasp_site_id].copy()

        #diff_palm_obj = absolute_palm_pos - state_dict['obj_pos'] + [0, 0.049, 0]
        #info['success'] = True if (np.linalg.norm(diff_palm_obj[:2]) < GRASP_THRESHOLD) else False

        if terminated or truncated:
            break
    
    #print(f"Episode {i+1}/{num_samples} - Success: {info['success']}")
    
    final_abs_palm = env.unwrapped.data.site_xpos[env.unwrapped.S_grasp_site_id].copy()
    init_abs_obj = init_state_dict['obj_pos'].copy()
    
    error_list.append(np.linalg.norm(final_abs_palm[:2] - init_abs_obj[:2] + [0, 0.049]))

    if error_list[-1] > 0.11:
        print(f"Epsisode {i}, Final error: {error_list[-1]:.4f}")

    env.close()

    if i % 100 == 0 and i > 0:
        print(f"Completed {i} episodes")#, success rate: {(sum(success_list) / (i+1)) * 100 :.2f}%")

print("Evaluation loop completed successfully.")