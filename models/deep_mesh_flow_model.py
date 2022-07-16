from tracemalloc import start
from resnet_features import resnet34
from typing import *
import torch.nn as nn
import torch
import torchgeometry
from torchgeometry.core.transformations import transform_points

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

mesh_estimator_body = resnet34()

class mesh_estimator_head(nn.Module):
    def __init__(self, mesh_grid_size:Tuple[int], upsample_size:Tuple[int], image_size:Tuple[int]) -> None:
        super().__init__()
        self.mesh_grid_size = mesh_grid_size
        self.upsample_size = upsample_size
        self.image_size = image_size
        # self.device = device
        fc_out_size = mesh_grid_size[0] * mesh_grid_size[1] * 2
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
        print("x shape:", x.shape)
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = out.view(out.shape[0], 2, self.mesh_grid_size[0], self.mesh_grid_size[1])
        if self.upsample_size != self.mesh_grid_size:
            unfold = nn.Unfold(kernel_size=(2, 2))
            mesh_flow = unfold(out).permute(0, 2, 1)
            mesh_flow = mesh_flow.reshape(mesh_flow.shape[0] * mesh_flow.shape[1], -1)
            mesh_flow = mesh_flow.reshape(-1, 2, 4).permute(0, 2, 1)
            # calculate origin mesh grid to calculate mesh homography matrices
            y_t = torch.matmul(torch.ones(self.mesh_grid_size[0], 1), torch.linspace(0, self.image_size[1], self.mesh_grid_size[1]).unsqueeze(0)).unsqueeze(2)
            x_t = torch.matmul(torch.linspace(0, self.image_size[0], self.mesh_grid_size[0]).unsqueeze(1), torch.ones(1, self.mesh_grid_size[1])).unsqueeze(2)
            # grid_sparse shape: [batch_size, 2, mesh_grid_size[0], mesh_grid_size[1]]
            grid_sparse = torch.cat([x_t, y_t], dim=2).permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            # unfolded grid sparse shape: [batch_size, 2 * 2 * 2, (mesh_grid_size[0] - 1) * (mesh_grid_size[1] - 1)]
            grid_sparse_unfold = unfold(grid_sparse).permute(0, 2, 1)
            # unfolded grid sparse shape: [batch_size * (mesh_grid_size[0] - 1) * (mesh_grid_size[1] - 1), 2 * 2 * 2]
            # caution: unfold tensor data format is [batch_size, c * kernel_h * kernel_w, patches]
            grid_sparse_unfold = grid_sparse_unfold.reshape(grid_sparse_unfold.shape[0] * grid_sparse_unfold.shape[1], -1)
            # unfolded grid sparse shape: [batch_size * (mesh_grid_size[0] - 1) * (mesh_grid_size[1] - 1), 4, 2]
            grid_sparse_unfold = grid_sparse_unfold.reshape(-1, 2, 4).permute(0, 2, 1)
            print("grid_sparse_unfold shape:" , grid_sparse_unfold.shape)
            print(grid_sparse_unfold.reshape(32, 2, 2, 2)[0, 0, 1, :], grid_sparse_unfold.reshape(32, 2, 2, 2)[1, 0, 0, :])
            print("After flow coordinates:")
            b = (grid_sparse_unfold + mesh_flow).reshape(32, 2, 2, 2)
            print(b[0, 0, 1, :], b[1, 0, 0, :])
            solved_matrices = torchgeometry.get_perspective_transform(grid_sparse_unfold, grid_sparse_unfold + mesh_flow)
            print("solved matrices:", solved_matrices.shape)
            # patch_width = self.image_size[1] / (self.mesh_grid_size[1] - 1)
            # patch_height = self.image_size[0] / (self.mesh_grid_size[0] - 1)
            # solved_matrices_scaled shape: [batch_size * (mesh_grid_size[0] - 1) * (mesh_grid_size[1] - 1), 3, 3]
            # solved_matrices_scaled = H_scale(solved_matrices, patch_width=patch_width, patch_height=patch_height, batch_size=solved_matrices.shape[0])

            # unfold dense grid, apply solved matrices to each patch, fold them into upsampled warped mesh, finally initial mesh is subtracted from warped mesh to obtain upsampled mesh flow
            # upsample_mesh (should be perspectively transformed), horizontal axis is y, vertical axis is x
            y_t = torch.matmul(torch.ones(self.upsample_size[0], 1), torch.linspace(0, self.image_size[1], self.upsample_size[1]).unsqueeze(0)).unsqueeze(2)
            x_t = torch.matmul(torch.linspace(0, self.image_size[0], self.upsample_size[0]).unsqueeze(1), torch.ones(1, self.upsample_size[1])).unsqueeze(2)
            # grid shape: [batch_size, 2, upsample_size[0], upsample_size[1]]
            grid_dense = torch.cat([x_t, y_t], dim=2).permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            unfold_kernel_size = ((self.upsample_size[0] - 1) // (self.mesh_grid_size[0] - 1) + 1, (self.upsample_size[1] - 1) // (self.mesh_grid_size[1] - 1) + 1)
            unfold_stride = ((self.upsample_size[0] - 1) // (self.mesh_grid_size[0] - 1), (self.upsample_size[1] - 1) // (self.mesh_grid_size[1] - 1))
            unfold2 = nn.Unfold(kernel_size=unfold_kernel_size, stride=unfold_stride)
            # unfold_grid_dense (direct result) shape: [batch_size, 2 * unfold_kernel_size[0]*[1], patches]
            unfold_grid_dense = unfold2(grid_dense).permute(0, 2, 1).reshape(solved_matrices.shape[0], -1)
            unfold_grid_dense = unfold_grid_dense.reshape(-1, 2, unfold_kernel_size[0] * unfold_kernel_size[1]).permute(0, 2, 1)
            print(unfold_grid_dense.shape)
            test = unfold_grid_dense.reshape(32, 5, 5, 2)
            print(test[0, 0, 4, :], test[1, 0, 0, :])
            # print(solved_matrices[0], print(solved_matrices[1]))
            warped_unfold_grid_dense = transform_points(solved_matrices, unfold_grid_dense)
            warped_unfold_grid_dense = warped_unfold_grid_dense.reshape(x.shape[0], -1, unfold_kernel_size[0], unfold_kernel_size[1], 2)
            warped_unfold_grid_dense = warped_unfold_grid_dense.permute(0, 4, 2, 3, 1).reshape(x.shape[0], 2 * unfold_kernel_size[0] * unfold_kernel_size[1], -1)
            # warped_unfold_grid_dense = warped_unfold_grid_dense.reshape(, -1, 2 * unfold_kernel_size[0] * unfold_kernel_size[1]).permute(0, 2, 1)
            # test
            a = warped_unfold_grid_dense.reshape(x.shape[0], 2, unfold_kernel_size[0], unfold_kernel_size[1], -1)
            print(a[0, :, 0, 4, 0], a[0, :, 0, 0, 1])
            print(a[0, :, 0, 0, 0])
            # fold is not the inverse operation of unfold. It will sum the values of overlapped region
            fold = torch.nn.Fold(output_size = (self.upsample_size[0], self.upsample_size[1]), kernel_size=unfold_kernel_size, stride=unfold_stride)
            warped_grid_dense = fold(warped_unfold_grid_dense)
            print(torch.arange(start=0, end=self.upsample_size[1], step=self.mesh_grid_size[1] - 1, dtype=torch.long))
            X, Y = torch.meshgrid(torch.arange(start=0, end=self.upsample_size[0], step=self.mesh_grid_size[0] - 1, dtype=torch.long),
                torch.arange(start=0, end=self.upsample_size[0], step=self.mesh_grid_size[0] - 1, dtype=torch.long),
                indexing="xy"
            )
            warped_grid_dense[:, :, X, Y] /= 2
            warped_grid_dense[:, :, 0, 0] *= 2
            warped_grid_dense[:, :, 0, -1] *= 2
            warped_grid_dense[:, :, -1, -1] *= 2
            warped_grid_dense[:, :, -1, 0] *= 2
            print(warped_grid_dense[0, :, 0, 0])
            print(warped_grid_dense[0, :, 0, 4])
            # print(torch.where(warped_grid_dense==a[0, 0, 0, 4, 0]))
            upsample_mesh_flow = warped_grid_dense - grid_dense
            out = upsample_mesh_flow
        
        return out

def H_scale(H:torch.Tensor, patch_width:float, patch_height:float, batch_size:int) -> torch.Tensor:
    M = torch.tensor([[patch_height / 2.0, 0., patch_height / 2.0],
                        [0., patch_width / 2.0, patch_width / 2.0],
                        [0., 0., 1.]], dtype=torch.float32)
    M_inv = torch.linalg.inv(M)
    M_tile = torch.tile(M.unsqueeze(0), dims=[batch_size, 1, 1])
    M_inv_tile = torch.tile(M_inv.unsqueeze(0), dims=[batch_size, 1, 1])
    scaled_H = torch.matmul(torch.matmul(M_inv_tile, H), M_tile)
    return scaled_H


if __name__ == "__main__":
    from torchsummary import summary
    model = mesh_estimator_head((5, 5), (17, 17), (128, 128))
    # a = torch.randn(10, 1024)
    # with torch.no_grad():
    #     print(model(a))
    print(summary(model, (1,1024), batch_size=10, device="cpu"))