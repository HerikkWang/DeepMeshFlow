from tracemalloc import start
from mesh_flow_upsample import mesh_flow_upsampling
from resnet_features import ResNet, resnet34, BasicBlock
from typing import *
import torch.nn as nn
import torch
import torchgeometry


class feature_extractor(nn.Module):
    def __init__(self, in_channels=1, out_channels=1) -> None:
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels, 4 * in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4 * in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(4 * in_channels, 8 * in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8 * in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(8 * in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        out = self.extractor(x)
        return out


class mask_predictor(nn.Module):
    def __init__(self, in_channels=1) -> None:
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Conv2d(in_channels, 4 * in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4 * in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(4 * in_channels, 8 * in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8 * in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(8 * in_channels, 16 * in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16 * in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(16 * in_channels, 32 * in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32 * in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(32 * in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        out = self.predictor(x)
        return out

class mesh_selector(nn.Module):
    def __init__(self, in_channels=2, out_shape=(3, 17, 17)) -> None:
        super().__init__()
        self.out_shape = out_shape
        self.out_feature_num = out_shape[0] * out_shape[1] * out_shape[2]
        self.features = ResNet(BasicBlock, [2, 2, 2, 2], in_channels)
        self.fc = nn.Linear(512, self.out_feature_num)
    
    def forward(self, x):
        out = self.features(x)
        out = self.fc(out)
        out = out.view(-1, self.out_shape[0], self.out_shape[1], self.out_shape[2])
        return out

mesh_estimator_body = resnet34(in_channels=2)

class mesh_estimator_head(nn.Module):
    def __init__(self, mesh_grid_size:Tuple[int], upsample_size:Tuple[int], image_size:Tuple[int]) -> None:
        super().__init__()
        self.mesh_grid_size = mesh_grid_size
        self.upsample_size = upsample_size
        self.image_size = image_size
        fc_out_size = mesh_grid_size[0] * mesh_grid_size[1] * 2
        self.fc1 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
        )
        # self.fc2 = nn.Sequential(
        #     nn.Linear(1024, 2048),
        #     nn.ReLU(),
        # )

        self.fc3 = nn.Sequential(
            nn.Linear(1024, fc_out_size),
        )
    
    def forward(self, x):
        out = self.fc1(x)
        # out = self.fc2(out)
        out = self.fc3(out)
        out = out.view(out.shape[0], 2, self.mesh_grid_size[0], self.mesh_grid_size[1])
        if self.upsample_size != self.mesh_grid_size:
            out = mesh_flow_upsampling(out, self.mesh_grid_size, self.upsample_size, self.image_size, x.shape[0])
        
        return out




if __name__ == "__main__":
    from torchsummary import summary
    model = mesh_estimator_head((5, 5), (17, 17), (128, 128))
    print(summary(model, (1,1024), batch_size=10, device="cpu"))