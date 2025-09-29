import torch
import numpy as np

def val(model, VAL_Y, VAL_C, time_len):
    error = 0.0
    for i in range(VAL_Y.shape[0]):
        cnmp_obs = np.concatenate((np.array([0.0]), np.zeros(VAL_Y.shape[2])), axis=-1).reshape((1, -1))
        context = VAL_C[i]
        action_trajectory = generate_trajectory(model, time_len, cnmp_obs, context)
        error += np.mean((VAL_Y[i] - np.array(action_trajectory)) ** 2)
    error /= VAL_Y.shape[0]
    return error

def generate_trajectory(model, time_len, obs, context):
    obs = torch.tensor(obs, dtype=torch.float64).unsqueeze(0)
    mask = torch.ones((1, obs.shape[1], obs.shape[1]), dtype=torch.float64)
    x_tar = torch.linspace(0, 1, time_len, dtype=torch.float64).unsqueeze(0).unsqueeze(-1)
    context = torch.tensor(context, dtype=torch.float64).unsqueeze(0)

    with torch.no_grad():
        output, _ = model(obs, context, mask, x_tar)
    mean, _ = output.chunk(2, dim=-1)
    return mean

        