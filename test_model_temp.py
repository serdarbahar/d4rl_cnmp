import minari
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from cnmp import CNMP, CNMP_H, CNMP_T, generate_trajectory, generate_action_temp
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull

# --- Model and Data Loading (Unchanged) ---
model = CNMP_T(d_x=1, d_y=30, d_SM=30).double()
model.load_state_dict(torch.load("best_model_newdataset_tempcontext.pth"))
model.eval()

dataset = minari.load_dataset('D4RL/relocate/human-v2')

# 3 subplots
import matplotlib.pyplot as plt
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

training_region = np.zeros((25, 6), dtype=np.float64)
for i in range(25):
    episode = dataset[i]
    obs = episode.observations
    training_region[i, :] = obs[0, -9:-3]
    axs[0].scatter(obs[0, -9], obs[0, -8], c='black', label='Hand to Ball')
    axs[1].scatter(obs[0, -7], obs[0, -6], c='black', label='Hand to Target')
    axs[2].scatter(obs[0, -5], obs[0, -4], c='black', label='Ball to Target')

# --- Corrected Evaluation Loop with Correct Joint Names and API ---
success_count = 0.0
for i in range(100):
    env = dataset.recover_environment(max_episode_steps=450, render_mode='human')
    init_obs, _ = env.reset(seed=420+i)

    obs_ = env.env.env.env._get_obs()

    # --- Trajectory Generation and Execution (Unchanged) ---
    cnmp_obs = np.array([np.concatenate((np.array([0.0]), init_obs[:30]), axis=-1)])
    
    context = np.zeros((12))  # Adjusted context size to match the model's expectation
    context[0:6] = init_obs[30:36]
    init_state_dict = env.env.env.env.get_env_state()
    context[6:9] = init_state_dict['obj_pos']
    context[9:12] = init_state_dict['target_pos']
    
    traj_length = 400
    obs = init_obs.copy()
    last_action = np.zeros(30)  # Initialize last action
    for step_num in range(450):
        if step_num < traj_length:
            temp_context = np.concat((context, obs[:30]), axis=-1)  # Adjusted context for temp action generation

            action = generate_action_temp(model, cnmp_obs, step_num, traj_length, temp_context)
            action = action.detach().numpy().reshape(-1)
            last_action = action 
        else:
            action = last_action
        obs, reward, terminated, truncated, info = env.step(action)
        

        if terminated or truncated:
            break
    

    if info['success']:
        axs[0].scatter(obs[-9], obs[-8], c='red', label='Hand to Ball')
        axs[1].scatter(obs[-7], obs[-6], c='red', label='Hand to Target')
        axs[2].scatter(obs[-5], obs[-4], c='red', label='Successful Test')
        success_count += 1.0
    else:
        axs[0].scatter(obs[-9], obs[-8], c='blue', label='Hand to Ball')
        axs[1].scatter(obs[-7], obs[-6], c='blue', label='Hand to Target')
        axs[2].scatter(obs[-5], obs[-4], c='blue', label='Failed Test')
    env.close()

    if i % 100 == 0:
        print(f"Completed {i} episodes, success rate: {(success_count / (i + 1)) * 100 :.2f}%")
    
for i in range(25):
    episode = dataset[i]
    obs = episode.observations
    axs[0].scatter(obs[0, -9], obs[0, -8], c='black', label='Hand to Ball')
    axs[1].scatter(obs[0, -7], obs[0, -6], c='black', label='Hand to Target')
    axs[2].scatter(obs[0, -5], obs[0, -4], c='black', label='Training Data')

axs[0].grid()
axs[1].grid()
axs[2].grid()

plt.savefig('evaluation_results_different_seed.png')

print(f"Success rate: {(success_count / 100.0) * 100:.2f}%")

print("Evaluation loop completed successfully.")