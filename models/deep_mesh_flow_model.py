from resnet_features import resnet34
from typing import *
import torch.nn as nn
import torch

class feature_extractor(nn.Module):
    def __init__(self, in_channel=1) -> None:
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channel, 4 * in_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4 * in_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(4 * in_channel, 8 * in_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8 * in_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(8 * in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        out = self.extractor(x)
        return out


class mask_predictor(nn.Module):
    def __init__(self, in_channel=1) -> None:
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Conv2d(in_channel, 4 * in_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4 * in_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(4 * in_channel, 8 * in_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8 * in_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(8 * in_channel, 16 * in_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16 * in_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(16 * in_channel, 32 * in_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32 * in_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(32 * in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        out = self.predictor(x)
        return out

mesh_estimator_body = resnet34()

class mesh_estimator_head(nn.Module):
    def __init__(self, mesh_grid_size:Tuple[int], upsample_size:Tuple[int], image_size:Tuple[int], device:torch.device) -> None:
        super().__init__()
        self.mesh_grid_size = mesh_grid_size
        self.upsample_size = upsample_size
        self.image_size = image_size
        self.device = device
        fc_out_size = (mesh_grid_size[0] + 1) * (mesh_grid_size[1] + 1) * 2
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(2048, fc_out_size),
        )
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = out.view(out.shape[0], 2, self.mesh_grid_size[0] + 1, self.mesh_grid_size[1] + 1)
        if self.upsample_size != self.mesh_grid_size:
            unfold = nn.Unfold(kernel_size=(2, 2))
            grid_out = unfold(out).permute(0, 2, 1)
            grid_out = grid_out.reshape(grid_out.shape[0] * grid_out.shape[1], -1)
            grid_out = grid_out.reshape(-1, 4, 2)
            # TODO: calculate origin mesh grid to calculate mesh homography matrices

            # upsample_mesh (should be perspectively transformed), horizontal axis is x, vertical axis is y
            x_t = torch.matmul(torch.ones(self.upsample_size[0], 1), torch.linspace(0, self.image_size[1], self.upsample_size[1]).unsqueeze(0)).unsqueeze(2)
            y_t = torch.matmul(torch.linspace(0, self.image_size[0], self.upsample_size[0]).unsqueeze(1), torch.ones(1, self.upsample_size[1])).unsqueeze(2)
            # grid shape: [batch_size, 2, upsample_size[0], upsample_size[1]]
            grid = torch.cat([x_t, y_t], dim=2).permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            # grid_unfold shape: [batch_size, 8, upsample_size[0] - 1, upsample_size[1] - 1]
            grid_unfold = unfold(grid)
            
        return out
