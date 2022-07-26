import torch
import torchgeometry
import torch.nn as nn
from typing import *
from torchgeometry.core.transformations import transform_points


def mesh_flow_upsampling(mesh_flow_tensor:torch.Tensor, mesh_grid_size:Tuple[int], upsample_grid_size:Tuple[int], img_size:Tuple[int], batch_size:int, device:torch.device):
    """upsample mesh flow result from mesh grid size to upsampling grid size

    Args:
      mesh_flow_tensor: tensor result of mesh flow. Data format: (N, 2, mesh_grid_size[0], mesh_grid_size[1])
      mesh_grid_size: mesh grid of low resolution
      upsample_grid_size: mesh grid of high resolutin
      img_size: size of input image
      batch_size: batch size of data
      device: torch device (cpu or gpu)
    
    Returns:
      upsampled mesh flow tensor with shape: (N, 2, upsample_grid_size[0], upsample_grid_size[1])
    """
    # step 1: unfold mesh flow tensor
    unfold = nn.Unfold(kernel_size=(2, 2))
    # caution: unfold tensor data format is [batch_size, c * kernel_h * kernel_w, patches]
    mesh_flow = unfold(mesh_flow_tensor).permute(0, 2, 1)
    mesh_flow = mesh_flow.reshape(mesh_flow.shape[0] * mesh_flow.shape[1], -1)
    mesh_flow = mesh_flow.reshape(-1, 2, 4).permute(0, 2, 1)
    # step 2: get mesh grid of mesh_grid size
    y_t = torch.matmul(torch.ones(mesh_grid_size[0], 1, dtype=torch.float, device=device), \
      torch.linspace(0, img_size[1], mesh_grid_size[1], dtype=torch.float, device=device).unsqueeze(0)).unsqueeze(2)
    x_t = torch.matmul(torch.linspace(0, img_size[0], mesh_grid_size[0], dtype=torch.float, device=device).unsqueeze(1), \
      torch.ones(1, mesh_grid_size[1], dtype=torch.float, device=device)).unsqueeze(2)
    # grid_sparse shape: [batch_size, 2, mesh_grid_size[0], mesh_grid_size[1]]
    grid_sparse = torch.cat([x_t, y_t], dim=2).permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
    # step 3: unfold sparse grid
    # unfolded grid sparse shape: [batch_size, 2 * 2 * 2, (mesh_grid_size[0] - 1) * (mesh_grid_size[1] - 1)]
    grid_sparse_unfold = unfold(grid_sparse).permute(0, 2, 1)
    # unfolded grid sparse shape: [batch_size * (mesh_grid_size[0] - 1) * (mesh_grid_size[1] - 1), 2 * 2 * 2]
    grid_sparse_unfold = grid_sparse_unfold.reshape(grid_sparse_unfold.shape[0] * grid_sparse_unfold.shape[1], -1)
    # unfolded grid sparse shape: [batch_size * (mesh_grid_size[0] - 1) * (mesh_grid_size[1] - 1), 4, 2]
    grid_sparse_unfold = grid_sparse_unfold.reshape(-1, 2, 4).permute(0, 2, 1)
    # step 4: use original mesh grid and mesh flow to calculate mesh upsampling homography matrices
    solved_matrices = torchgeometry.get_perspective_transform(grid_sparse_unfold, grid_sparse_unfold + mesh_flow)

    # Whether Scaling?
    # patch_width = img_size[1] / (mesh_grid_size[1] - 1)
    # patch_height = img_size[0] / (mesh_grid_size[0] - 1)
    # solved_matrices_scaled = H_scale(solved_matrices, patch_width=patch_width, patch_height=patch_height, batch_size=batch_size)

    # step 5: get dense grid of upsample_grid_size
    y_t = torch.matmul(torch.ones(upsample_grid_size[0], 1, dtype=torch.float, device=device), \
      torch.linspace(0, img_size[1], upsample_grid_size[1], dtype=torch.float, device=device).unsqueeze(0)).unsqueeze(2)
    x_t = torch.matmul(torch.linspace(0, img_size[0], upsample_grid_size[0], dtype=torch.float, device=device).unsqueeze(1), \
      torch.ones(1, upsample_grid_size[1], dtype=torch.float, device=device)).unsqueeze(2)
    # grid shape: [batch_size, 2, upsample_grid_size[0], upsample_grid_size[1]]
    grid_dense = torch.cat([x_t, y_t], dim=2).permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
    # step 6: unfold dense grid
    unfold_kernel_size = ((upsample_grid_size[0] - 1) // (mesh_grid_size[0] - 1) + 1, (upsample_grid_size[1] - 1) // (mesh_grid_size[1] - 1) + 1)
    unfold_stride = ((upsample_grid_size[0] - 1) // (mesh_grid_size[0] - 1), (upsample_grid_size[1] - 1) // (mesh_grid_size[1] - 1))
    unfold2 = nn.Unfold(kernel_size=unfold_kernel_size, stride=unfold_stride)
    # unfold_grid_dense (direct result) shape: [batch_size, 2 * unfold_kernel_size[0]*[1], patches]
    unfold_grid_dense = unfold2(grid_dense).permute(0, 2, 1).reshape(solved_matrices.shape[0], -1)
    unfold_grid_dense = unfold_grid_dense.reshape(-1, 2, unfold_kernel_size[0] * unfold_kernel_size[1]).permute(0, 2, 1)
    # step 7: transform points in unfolded dense grid using homography matrices solved from sparse mesh
    warped_unfold_grid_dense = transform_points(solved_matrices, unfold_grid_dense)
    warped_unfold_grid_dense = warped_unfold_grid_dense.reshape(batch_size, -1, unfold_kernel_size[0], unfold_kernel_size[1], 2)
    warped_unfold_grid_dense = warped_unfold_grid_dense.permute(0, 4, 2, 3, 1).reshape(batch_size, 2 * unfold_kernel_size[0] * unfold_kernel_size[1], -1)
    
    # Caution: fold is not the inverse operation of unfold. It will sum the values of overlapped region
    # step 8: fold back transform points to mesh grid
    fold = torch.nn.Fold(output_size = (upsample_grid_size[0], upsample_grid_size[1]), kernel_size=unfold_kernel_size, stride=unfold_stride)
    warped_grid_dense = fold(warped_unfold_grid_dense)
    X_border = torch.arange(start=0, end=upsample_grid_size[0] - 1, step=unfold_kernel_size[0] - 1, dtype=torch.long)[1:]
    Y_border = torch.arange(start=0, end=upsample_grid_size[1] - 1, step=unfold_kernel_size[1] - 1, dtype=torch.long)[1:]
    X1, Y1 = torch.meshgrid(torch.arange(upsample_grid_size[0], dtype=torch.long), Y_border, indexing="ij")
    X2, Y2 = torch.meshgrid(X_border, torch.arange(upsample_grid_size[1], dtype=torch.long), indexing="ij")
    warped_grid_dense[:, :, X1, Y1] /= 2
    warped_grid_dense[:, :, X2, Y2] /= 2

    # step 9: subtract origin dense mesh grid from transformed mesh grid to obtain upsampled mesh flow
    upsample_mesh_flow = warped_grid_dense - grid_dense
    out = upsample_mesh_flow

    return out

if __name__ == "__main__":
    # mesh_flow = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.float).permute(1, 0).reshape(2, 2, -1).unsqueeze(0)
    # # print(mesh_flow[0, :, 0, 0])
    # b = mesh_flow_upsampling(mesh_flow, (2, 2), (17, 17), (128, 128), 1, torch.device("cpu"))
    # print(b)
    import sys
    sys.path.append("..")
    from utils.solve_DLT import *
    from PIL import Image
    from torchvision import transforms
    im_in = Image.open("../utils/test.jpg")
    im_tensor = transforms.ToTensor()(im_in).unsqueeze(0)
    mesh_flow = torch.meshgrid(torch.arange(5, dtype=torch.float) - 1, \
        torch.arange(5, dtype=torch.float) - 1, indexing='ij')
    mesh_flow = torch.cat([mesh_flow[0].unsqueeze(0), mesh_flow[1].unsqueeze(0)], dim=0).unsqueeze(0)
    print(mesh_flow.shape)
    b = mesh_flow_upsampling(mesh_flow, (5, 5), (9, 9), (128, 128), 1, torch.device("cpu"))
    c = solve_mesh_flow_DLT(b, torch.device("cpu"), (16, 16), (128 ,128))
    img_tensor = spatial_transform_by_grid(im_tensor, c[0], torch.device("cpu"))
    im_out = transforms.ToPILImage()(img_tensor.squeeze(0))
    im_out.save("../res.jpg")
    # print(torch.where(b.squeeze(0) > 10))
    # a = torch.randn(size=(1, 3, 100, 100))
    # a_unfold = nn.Unfold(kernel_size=(3, 3))(a)
    # a_foldback = nn.Fold((100, 100), (3, 3))(a_unfold)
    # print(torch.abs(a - a_foldback).sum())