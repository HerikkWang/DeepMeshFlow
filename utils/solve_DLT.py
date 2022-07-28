from typing import Tuple
import torch.nn.functional as F
import torchgeometry
import torch.nn as nn
import torch

def solve_mesh_flow_DLT(mesh_flow: torch.Tensor, device: torch.device, patch_size: Tuple[int], image_size: Tuple[int]) -> Tuple[torch.Tensor]:
    batch_size = mesh_flow.shape[0]
    unfold_mesh_flow = grid2pointgroup(mesh_flow, (2, 2))
    mesh_grid_X, mesh_grid_Y = torch.meshgrid(torch.arange(start=0, end=image_size[0] + 1, step=patch_size[0], dtype=torch.float, device=device),
        torch.arange(start=0, end=image_size[1] + 1, step=patch_size[1], dtype=torch.float, device=device,),
        indexing="ij",
    )
    mesh_grid = torch.cat([mesh_grid_X.unsqueeze(0), mesh_grid_Y.unsqueeze(0)], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)
    unfold_mesh_grid = grid2pointgroup(mesh_grid, (2, 2))
    solved_matrices = torchgeometry.get_perspective_transform(unfold_mesh_grid, unfold_mesh_flow + unfold_mesh_grid)
    # solved_matrices = H_scale(solved_matrices, patch_size[0], patch_size[1], unfold_mesh_grid.shape[0], device=device)

    # warp image grid points
    points_grid = torch.meshgrid(torch.arange(image_size[0], dtype=torch.float, device=device), \
        torch.arange(image_size[1], dtype=torch.float, device=device), indexing='ij')
    points_grid = torch.cat([points_grid[0].unsqueeze(0), points_grid[1].unsqueeze(0)], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)
    grid_unfolder = nn.Unfold(kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1]))
    # unfold_points_grid shape: [N, 2 * patch_size * patch_size, grid_size * grid_size]
    unfold_points_grid = grid_unfolder(points_grid)
    # after reshape and permutation: [N * grid_size * grid_size, patch_size * patch_size, 2]
    unfold_points_grid = unfold_points_grid.permute(0, 2, 1).reshape(-1, 2 * patch_size[0] * patch_size[1]).reshape(-1, 2, patch_size[0] * patch_size[1]).permute(0, 2, 1)
    warped_unfold_points_grid = torchgeometry.core.transform_points(solved_matrices, unfold_points_grid)
    print("warped_unfold_points_grid:", warped_unfold_points_grid[3])

    # FOLD BACK
    warped_unfold_points_grid = warped_unfold_points_grid.permute(0, 2, 1).reshape(-1, 2 * patch_size[0] * patch_size[1]). \
        reshape(batch_size, -1, 2 * patch_size[0] * patch_size[1]).permute(0, 2, 1)
    folder = nn.Fold(output_size=image_size, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1]))
    warped_points_grid = folder(warped_unfold_points_grid)
    # print(warped_points_grid)

    return warped_points_grid, solved_matrices

def grid2pointgroup(grid:torch.Tensor, kernel_size:Tuple[int]) -> torch.Tensor:
    """Unfold grid for solving homograhpy matrices
    """
    unfolder = nn.Unfold(kernel_size=kernel_size)
    grid_unfold = unfolder(grid)
    grid_unfold = grid_unfold.permute(0, 2, 1).reshape(-1, 8).reshape(-1, 2, 4).permute(0, 2, 1)

    return grid_unfold

def H_scale(H:torch.Tensor, patch_width:float, patch_height:float, batch_size:int, device:torch.device) -> torch.Tensor:
    M = torch.tensor([[patch_height / 2.0, 0., patch_height / 2.0],
                        [0., patch_width / 2.0, patch_width / 2.0],
                        [0., 0., 1.]], dtype=torch.float32, device=device)
    M_inv = torch.linalg.inv(M)
    M_tile = torch.tile(M.unsqueeze(0), dims=[batch_size, 1, 1])
    M_inv_tile = torch.tile(M_inv.unsqueeze(0), dims=[batch_size, 1, 1])
    scaled_H = torch.matmul(torch.matmul(M_inv_tile, H), M_tile)
    return scaled_H

def spatial_transform_by_grid(img:torch.Tensor, grid:torch.Tensor, device:torch.device) -> torch.Tensor:
    # padding the image by 1
    # img = F.pad(img, (1, 1, 1, 1), "constant", 0)
    batch_size, channels, height, width = img.shape
    # grid += 1
    # grid = F.pad(grid, (1, 1, 1, 1), "constant", 0)
    grid = grid.reshape(batch_size, 2, -1)
    x = grid[:, 0, :].reshape(-1)
    y = grid[:, 1, :].reshape(-1)
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, height - 1)
    x1 = torch.clamp(x1, 0, height - 1)
    y0 = torch.clamp(y0, 0, width - 1)
    y1 = torch.clamp(y1, 0, width - 1)

    base = (torch.arange(batch_size, device=device, dtype=torch.float) * width * height).reshape(-1, 1)
    base = torch.matmul(base, torch.ones((1, height * width), dtype=torch.float, device=device)).view(-1).long()
    base_x0 = base + x0 * width
    base_x1 = base + x1 * width
    idx_a = base_x0 + y0
    idx_b = base_x1 + y0
    idx_c = base_x0 + y1
    idx_d = base_x1 + y1

    im_flat = img.permute(0, 2, 3, 1).reshape(-1, channels).float()
    Ia, Ib, Ic, Id = im_flat[idx_a], im_flat[idx_b], im_flat[idx_c], im_flat[idx_d]

    x0_f = x0.float()
    x1_f = x1.float()
    y0_f = y0.float()
    y1_f = y1.float()
    # test
    
    wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
    wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
    wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
    wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
    # print(wa, wb, wc, wd)

    output = wa * Ia + wb * Ib + wc * Ic + wd * Id
    output = output.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
    # output = output[:, :, 1:-1, 1:-1]

    return output 

# Test
if __name__ == "__main__":
    from PIL import Image
    from torchvision import transforms
    mesh_flow = torch.randn(8, 2, 17, 17)
    # mesh_flow = torch.tensor([(1, 1), (1, 1), (10, 1), (10, 1)], dtype=torch.float).permute(1, 0).reshape(2, 2, 2).unsqueeze(0)
    mesh_flow = torch.tensor([(0, 0), (0, 0), (0, 0), (0, 0), (0, 30), (0, 0), (0, 0), (0, 0), (0, 0)], dtype=torch.float).permute(1, 0).reshape(2, 3, 3).unsqueeze(0)
    # print(mesh_flow)
    # mesh_flow = torch.tensor([(0, 0), (0, 0), (-5, 1), (5, 1)], dtype=torch.float).permute(1, 0).reshape(2, 2, 2).unsqueeze(0)

    b, c = solve_mesh_flow_DLT(mesh_flow, torch.device("cpu"), (64, 64), (128, 128))
    # print(b)
    img = Image.open("../im_test/raw.jpg").convert("RGB")
    input_tensor = transforms.ToTensor()(img).unsqueeze(0)
    warp_tensor = spatial_transform_by_grid(input_tensor, b, torch.device("cpu"))
    warp_img = transforms.ToPILImage()(warp_tensor.squeeze(0))
    warp_img.save("../im_test/warp_raw.jpg")
    # print(torch.where(warp_mask < 0))
    # a = torch.arange(16).reshape(2, 2, 2, 2)
    # b = a + 1
    # ab = torch.cat([a.unsqueeze(1), b.unsqueeze(1)], dim=1)
    # c = torch.randn(size=(2, 1, 2, ))

