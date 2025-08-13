import torch
import torch.nn as nn

class CNMP(nn.Module):
    def __init__(self, d_x, d_y, d_SM):
        super(CNMP, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(d_x + d_y, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256)
        )

        self.decoder = nn.Sequential(
            nn.Linear(d_x + 9 + 256, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.1), 
            nn.Linear(512, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 2 * d_SM)  # Output mean and std for
        )
    def forward(self, obs, context, mask, x_tar): # obs is (n, d_x + d_y)

        r = self.encoder(obs)
        masked_r = torch.bmm(mask, r)
        masked_r_sum = torch.sum(masked_r, dim=1, keepdim=True)  # (1, 128)
        r_avg = masked_r_sum / torch.sum(mask, dim=[1,2], keepdim=True)  # (1, 128)
        r_avg = r_avg.repeat(1, x_tar.shape[1], 1)
        context = context.unsqueeze(1).repeat(1, x_tar.shape[1], 1)  # (n, 1, 9)
        concat = torch.cat((r_avg, context, x_tar), dim=-1)
        output = self.decoder(concat) # (2*d_y,)
        return output, r_avg
    
class CNMP_H(nn.Module):
    def __init__(self, d_x, d_y, d_SM):
        super(CNMP_H, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(d_x + d_y, 64), nn.LayerNorm(64), nn.ReLU(), 
            nn.Linear(64, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256),
        )

        self.decoder = nn.Sequential(
            nn.Linear(d_x + 15 + 256, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 2 * d_SM)  # Output mean and std for
        )
    def forward(self, obs, context, mask, x_tar): # obs is (n, d_x + d_y)

        r = self.encoder(obs)
        masked_r = torch.bmm(mask, r)
        masked_r_sum = torch.sum(masked_r, dim=1, keepdim=True)  # (1, 128)
        r_avg = masked_r_sum / torch.sum(mask, dim=[1,2], keepdim=True)  # (1, 128)
        r_avg = r_avg.repeat(1, x_tar.shape[1], 1)
        context = context.unsqueeze(1).repeat(1, x_tar.shape[1], 1)  # (n, 1, 9)
        concat = torch.cat((r_avg, context, x_tar), dim=-1)
        output = self.decoder(concat) # (2*d_y,)
        return output, r_avg
    
def generate_trajectory(model, obs, context):
    obs = torch.tensor(obs, dtype=torch.float64).unsqueeze(0)
    mask = torch.ones((1, obs.shape[1], obs.shape[1]), dtype=torch.float64)
    x_tar = torch.linspace(0, 1, 200, dtype=torch.float64).unsqueeze(0).unsqueeze(-1)
    context = torch.tensor(context, dtype=torch.float64).unsqueeze(0)
    with torch.no_grad():
        output, _ = model(obs, context, mask, x_tar)
    mean, _ = output.chunk(2, dim=-1)
    return mean

class CNMP_v2(nn.Module):
    def __init__(self, d_x, d_y, d_SM):
        super(CNMP_v2, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(d_x + d_y, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.ReLU(), 
            nn.Linear(512, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256)
        )

        self.decoder = nn.Sequential(
            nn.Linear(d_x + 0 + 256, 512), nn.LayerNorm(512), nn.ReLU(), 
            nn.Linear(512, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 2 * d_SM)  # Output mean and std for
        )
    def forward(self, obs, context, mask, x_tar): # obs is (n, d_x + d_y)

        r = self.encoder(obs)
        masked_r = torch.bmm(mask, r)
        masked_r_sum = torch.sum(masked_r, dim=1, keepdim=True)  # (1, 128)
        r_avg = masked_r_sum / torch.sum(mask, dim=[1,2], keepdim=True)  # (1, 128)
        r_avg = r_avg.repeat(1, x_tar.shape[1], 1)
        #context = context.unsqueeze(1).repeat(1, x_tar.shape[1], 1)  # (n, 1, 9)
        #concat = torch.cat((r_avg, context, x_tar), dim=-1)
        concat = torch.cat((r_avg, x_tar), dim=-1)
        output = self.decoder(concat) # (2*d_y,)
        return output, r_avg

class CNMP_T(nn.Module):
    def __init__(self, d_x, d_y, d_SM):
        super(CNMP_T, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(d_x + d_y, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256),
        )

        self.decoder = nn.Sequential(
            nn.Linear(d_x + (12 + 30) + 256, 1024), nn.LayerNorm(1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 2 * d_SM)  # Output mean and std for
        )
    def forward(self, obs, context, mask, x_tar): # obs is (n, d_x + d_y)
        r = self.encoder(obs)
        masked_r = torch.bmm(mask, r)
        masked_r_sum = torch.sum(masked_r, dim=1, keepdim=True)  # (1, 128)
        r_avg = masked_r_sum / torch.sum(mask, dim=[1,2], keepdim=True)  # (1, 128)
        r_avg = r_avg.repeat(1, x_tar.shape[1], 1)
        context = context.unsqueeze(1).repeat(1, x_tar.shape[1], 1)  # (n, 1, 9)
        concat = torch.cat((r_avg, context, x_tar), dim=-1)
        #concat = torch.cat((r_avg, x_tar), dim=-1)
        output = self.decoder(concat) # (2*d_y,)
        return output, r_avg

def generate_action_temp(model, obs, i, traj_len, temp_context):
    obs = torch.tensor(obs, dtype=torch.float64).unsqueeze(0)  # (1, 1, d_x + d_y)
    mask = torch.ones((1, obs.shape[1], obs.shape[1]), dtype=torch.float64)
    temp_context = torch.tensor(temp_context, dtype=torch.float64).unsqueeze(0)  # (1, 9)
    x_tar = torch.linspace(0, 1, traj_len).unsqueeze(0).unsqueeze(-1)
    output = torch.zeros((2*30))
    with torch.no_grad():
            output, _ = model(obs, temp_context, mask, x_tar[:, i:i+1, :])
    mean, _ = output.chunk(2, dim=-1)
    return mean

class CNMP_SA(nn.Module):
    def __init__(self, d_x, d_y, d_SM):
        super(CNMP_SA, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(d_x + d_y, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256),
        )

        self.decoder = nn.Sequential(
            nn.Linear(d_x + 256, 1024), nn.LayerNorm(1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 2 * d_SM)  # Output mean and std for
        )
    def forward(self, obs, mask, x_tar): # obs is (n, d_x + d_y)

        r = self.encoder(obs)
        masked_r = torch.bmm(mask, r)
        masked_r_sum = torch.sum(masked_r, dim=1, keepdim=True)  # (1, 128)
        r_avg = masked_r_sum / torch.sum(mask, dim=[1,2], keepdim=True)  # (1, 128)
        r_avg = r_avg.repeat(1, x_tar.shape[1], 1)
        concat = torch.cat((r_avg, x_tar), dim=-1)
        output = self.decoder(concat) # (2*d_y,)
        return output, r_avg

def generate_trajectory_sa(model, obs):
    obs = torch.tensor(obs, dtype=torch.float64).unsqueeze(0)  # (1, 1, d_x + d_y)
    mask = torch.ones((1, obs.shape[1], obs.shape[1]), dtype=torch.float64)
    x_tar = torch.linspace(0, 1, 200).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        output, _ = model(obs, mask, x_tar)
    mean, _ = output.chunk(2, dim=-1)
    return mean