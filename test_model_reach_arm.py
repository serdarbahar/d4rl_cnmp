import minari
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from cnmp_ import CNMP_H, generate_trajectory # Assuming these are your custom modules
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from utils import sample_from_hull

# --- Model and Data Loading (Unchanged) ---
model = CNMP_H(d_x=1, d_y=5, d_SM=5).double()
model.load_state_dict(torch.load("save/best_models_reach_arm_v1/model_47600.pth"))
model.eval()

dataset = minari.load_dataset('D4RL/relocate/human-v2')
normalization_values = {"actions_min": [], 
                        "actions_max": [],
                        "context_min": [],
                        "context_max": []}

time_len = 451

action_data = np.load('data/reach_arm_actions_v1.npy')  # shape (25, 451, 30)
observation_data = np.load('data/reach_arm_observations_v1.npy')  # shape (25, 451, 42)
num_data = action_data.shape[0]

Y = np.zeros((num_data, time_len, 5))
Y[:, 1:] = action_data[:num_data]
C = np.zeros((num_data, 15))
for i in range(num_data):
    C[i, :9] = observation_data[i, 0, 30:39]
    C[i, 9:] = observation_data[i, 0, 42:]  # add the first observation as context

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

normalized_training_contexts = C.copy()
for dim in range(C.shape[-1]):
    normalized_training_contexts[:, dim] = (C[:, dim] - normalization_values['context_min'][dim]) / \
                                           (normalization_values['context_max'][dim] - normalization_values['context_min'][dim] + 1e-8)
    
convex_hull = ConvexHull(C[:, [9, 10]].copy())

num_samples = 1000

random_samples = sample_from_hull(convex_hull, num_samples)
#random_samples = C[:, [9, 10]].copy() 

test_contexts = []
# --- Corrected Evaluation Loop with Correct Joint Names and API ---
success_list = []
error_list = []
for i in range(num_samples):
    env = dataset.recover_environment(max_episode_steps=500, render_mode='human')
    init_obs, _ = env.reset(seed=420+i)
    env = env.unwrapped

    context_from_hull = random_samples[i]

    full_init_state = env.get_env_state()

    new_state_dict = {
        'qpos': full_init_state['qpos'].copy(),
        'qvel': full_init_state['qvel'].copy(),
        'target_pos': full_init_state['target_pos'].copy(),
        'obj_pos': np.array([context_from_hull[0], context_from_hull[1], 0.035]),
    }
    env.set_env_state(new_state_dict)

    cnmp_normalized = Y[0, 0, :].copy()
    for dim in range(cnmp_normalized.shape[-1]):
        cnmp_normalized[dim] = (cnmp_normalized[dim] - normalization_values['actions_min'][dim]) / \
                               (normalization_values['actions_max'][dim] - normalization_values['actions_min'][dim] + 1e-8)

    # --- Trajectory Generation and Execution (Unchanged) ---
    cnmp_obs = np.array([np.concatenate((np.array([0.0]), cnmp_normalized), axis=-1)])

    context = np.zeros((15))  # Adjusted context size to match the model's expectation
    init_state_dict = env.get_env_state()
    context[0:9] = env._get_obs()[30:39].copy()
    context[9:12] = init_state_dict['obj_pos'].copy()
    context[12:15] = init_state_dict['target_pos'].copy()

    for dim in range(15):
        context[dim] = (context[dim] - normalization_values['context_min'][dim]) / \
                       (normalization_values['context_max'][dim] - normalization_values['context_min'][dim] + 1e-8)
        #print("%.3f" % context[dim].item(), end=' ')

    test_contexts.append(context)

    traj_length = 450
    action_trajectory = generate_trajectory(model, cnmp_obs, context).squeeze(0)
    action_trajectory = action_trajectory.detach().numpy()
    action_trajectory = interp1d(np.linspace(0, 1, len(action_trajectory)), action_trajectory, axis=0)(np.linspace(0, 1, traj_length))

    for dim in range(action_trajectory.shape[-1]):
        action_trajectory[:, dim] = (action_trajectory[:, dim] * 
                                     (normalization_values['actions_max'][dim] - normalization_values['actions_min'][dim]) + 
                                     normalization_values['actions_min'][dim])

    action = np.zeros((30,))
    for step_num in range(450):
        action[:5] = action_trajectory[step_num] if step_num < traj_length else action_trajectory[-1]
        obs, reward, terminated, truncated, info = env.step(action)
        info['success'] = True if (np.linalg.norm(obs[30:31]) < 0.055 and obs[32] < 0.025) else False
        if terminated or truncated:
            break
            
    success_list.append(info['success']* 1.0)
    
    final_abs_palm = env.data.site_xpos[env.S_grasp_site_id].copy()
    init_abs_obs = init_state_dict['obj_pos'].copy()
    
    error_list.append(np.linalg.norm(final_abs_palm - init_abs_obs))
    #print(error_list[-1])

    env.close()

    if i % 100 == 0:
        print(f"Completed {i} episodes, success rate: {(sum(success_list) / (i+1)) * 100 :.2f}%")

print(f"Success rate: {(sum(success_list) / len(success_list)) * 100:.2f}%")


test_contexts = np.array(test_contexts)

# Separate successful and failed test contexts for plotting
success_mask = np.array(success_list, dtype=bool)
failed_mask = ~success_mask

# plot all dimensions of training and test contexts, each 2 dimensions in a scatter plot. there will be 7 plots in total
import matplotlib.pyplot as plt

dims = {0: (9, 10)}
dim_labels = {0: {'xlabel': 'Ball X Position (Absolute)', 'ylabel': 'Ball Y Position (Absolute)'}}


"""
plt.figure(figsize=(8, 5))
for i in range(1):
    dim1, dim2 = dims[i]
    plt.subplot(1, 1, i + 1)
    plt.scatter(training_contexts[:, dim1], training_contexts[:, dim2], color='black', label='Training Points')

    plt.scatter(test_contexts[failed_mask, dim1], test_contexts[failed_mask, dim2], color='red', alpha=0.2, label='Fail')
    plt.scatter(test_contexts[success_mask, dim1], test_contexts[success_mask, dim2], color='green', alpha=0.2, label='Success')
    
    plt.xlabel(dim_labels[i]['xlabel'], fontsize=12)
    plt.ylabel(dim_labels[i]['ylabel'], fontsize=12)

    plt.grid(alpha=0.3)

    plt.xlim(-0.25, 1.25)
    plt.ylim(-0.25, 1.25)

    for simplex in convex_hull.simplices:
        plt.plot(training_contexts[simplex, dim1], training_contexts[simplex, dim2], color='blue', alpha=0.5)

    

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig('reach_success.png')
plt.show()
"""

skip_indices = [0, 1, 13, 19, 20, 21]
plt.figure(figsize=(8, 5))
# plot color map according to error_list, generate cmap
cmap = plt.cm.get_cmap('coolwarm')
norm = plt.Normalize(vmin=0.0, vmax=0.20)
for i in range(1):
    dim1, dim2 = dims[i]
    plt.subplot(1, 1, i + 1)

    skip_count = 0
    for j in range(25):

        if j in skip_indices:
            skip_count += 1
            continue

        plt.scatter(normalized_training_contexts[:, dim1], normalized_training_contexts[:, dim2], color='black', label='Training Points')
        # add a text, the number of the point
        plt.text(normalized_training_contexts[j - skip_count, dim1], normalized_training_contexts[j - skip_count, dim2], str(j), fontsize=8, color='black', ha='right', va='bottom')

    plt.xlabel(dim_labels[i]['xlabel'], fontsize=12)
    plt.ylabel(dim_labels[i]['ylabel'], fontsize=12)

    plt.grid(alpha=0.3)

    # plot color map according to error_list
    scatter = plt.scatter(test_contexts[:, dim1], test_contexts[:, dim2], c=error_list, cmap=cmap, norm=norm, alpha=0.6)
    plt.colorbar(scatter, label='Error (Norm)')

    plt.xlim(-0.25, 1.25)
    plt.ylim(-0.25, 1.25)

    for simplex in convex_hull.simplices:
        plt.plot(normalized_training_contexts[simplex, dim1], normalized_training_contexts[simplex, dim2], color='blue', alpha=0.5)

#plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig('reach_error_v1.png')
plt.show()

print("Evaluation loop completed successfully.")