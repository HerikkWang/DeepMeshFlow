from typing import Tuple
import torch.nn.functional as F
import torchgeometry
import torch.nn as nn
import torch

def solve_mesh_flow_DLT(mesh_flow: torch.Tensor, device: torch.device, patch_size: Tuple[int], image_size: Tuple[int]) -> torch.Tensor:
    batch_size = mesh_flow.shape[0]
    unfold_mesh_flow = grid2pointgroup(mesh_flow, (2, 2))
    mesh_grid_X, mesh_grid_Y = torch.meshgrid(torch.arange(start=0, end=image_size[0] + 1, step=patch_size[0], dtype=torch.long, device=device),
        torch.arange(start=0, end=image_size[1] + 1, step=patch_size[1], dtype=torch.long, device=device,),
        indexing="xy",
    )
    mesh_grid = torch.cat([mesh_grid_X.unsqueeze(0), mesh_grid_Y.unsqueeze(0)], dim=0).unsqueeze(0).expand(batch_size, 1, 1, 1)
    unfold_mesh_grid = grid2pointgroup(mesh_grid, (2, 2))
    solved_matrices = torchgeometry.get_perspective_transform(unfold_mesh_grid, unfold_mesh_flow + unfold_mesh_grid)
    solved_matrices = H_scale(solved_matrices, patch_size[0], patch_size[1], batch_size)

    # warp image grid points
    points_grid = torch.meshgrid(torch.arange(image_size[0]), torch.arange(image_size[1]), indexing='xy')
    points_grid = torch.cat([points_grid[0].unsqueeze(0), points_grid[1].unsqueeze(0)], dim=0).unsqueeze(0).expand(batch_size, 1, 1, 1)
    grid_unfolder = nn.Unfold(kernel_size=(image_size[0] // patch_size[0], image_size[1] // patch_size[1]), 
        stride=(image_size[0] // patch_size[0], image_size[1] // patch_size[1])
    )
    # unfold_points_grid shape: [N, 2 * patch_size * patch_size, grid_size * grid_size]
    unfold_points_grid = grid_unfolder(points_grid)
    # after reshape and permutation: [N * grid_size * grid_size, patch_size * patch_size, 2]
    unfold_points_grid = unfold_points_grid.permute(0, 2, 1).reshape(-1, 2 * patch_size * patch_size).reshape(-1, 2, patch_size * patch_size).permute(0, 2, 1)
    warped_unfold_points_grid = torchgeometry.core.transform_points(solved_matrices)

    # FOLD BACK
    warped_unfold_points_grid = warped_unfold_points_grid.permute(0, 2, 1).reshape(-1, 2 * patch_size * patch_size).reshape(batch_size, -1, 2 * patch_size * patch_size) \
        .permute(0, 2, 1)
    folder = nn.Fold(output_size=image_size, kernel_size=(image_size[0] // patch_size[0], image_size[1] // patch_size[1]), 
        stride=(image_size[0] // patch_size[0], image_size[1] // patch_size[1]))
    warped_points_grid = folder(warped_unfold_points_grid)

    return warped_points_grid


def grid2pointgroup(grid:torch.Tensor, kernel_size:Tuple[int]) -> torch.Tensor:
    """Unfold grid for solving homograhpy matrices
    """
    unfolder = nn.Unfold(kernel_size=kernel_size)
    grid_unfold = unfolder(grid)
    grid_unfold = grid_unfold.permute(0, 2, 1).reshape(-1, 8).reshape(-1, 2, 4).permute(0, 1, 2)

    return grid_unfold

def H_scale(H:torch.Tensor, patch_width:float, patch_height:float, batch_size:int) -> torch.Tensor:
    M = torch.tensor([[patch_height / 2.0, 0., patch_height / 2.0],
                        [0., patch_width / 2.0, patch_width / 2.0],
                        [0., 0., 1.]], dtype=torch.float32)
    M_inv = torch.linalg.inv(M)
    M_tile = torch.tile(M.unsqueeze(0), dims=[batch_size, 1, 1])
    M_inv_tile = torch.tile(M_inv.unsqueeze(0), dims=[batch_size, 1, 1])
    scaled_H = torch.matmul(torch.matmul(M_inv_tile, H), M_tile)
    return scaled_H

def spatial_transform_by_grid(img:torch.Tensor, grid:torch.Tensor, device:torch.device) -> torch.Tensor:
    # padding the image by 1
    img = F.pad(img, (1, 1, 1, 1), "constant", 0)
    batch_size, channels, height, width = img.shape
    grid += 1
    grid = F.pad(grid, (1, 1, 1, 1), "constant", 0)
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
    print(x0_f[-1], x1_f[-1], y0_f[-1], y1_f[-1])
    print(x[-1], y[-1])
    
    wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
    wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
    wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
    wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
    # print(wa, wb, wc, wd)

    output = wa * Ia + wb * Ib + wc * Ic + wd * Id
    output = output.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
    output = output[:, :, 1:-1, 1:-1]

    return output 

# Test
if __name__ == "__main__":
    from PIL import Image
    from torchvision import transforms
    import cv2
    import numpy as np
    im = Image.open("/home/wyq/DeepImageStitching-pytorch/align_data/train/input1/000001.jpg")
    im.save("raw.jpg")
    im_t = (transforms.ToTensor()(im)).unsqueeze(0)
    # im_t = F.pad(im_t, (1, 1, 1, 1), 'constant', 0)
    points_grid = torch.meshgrid(torch.arange(128, dtype=torch.float), torch.arange(128, dtype=torch.float), indexing='ij')
    points_grid = torch.cat([points_grid[0].unsqueeze(0), points_grid[1].unsqueeze(0)], dim=0).unsqueeze(0)
    device = torch.device("cpu")
    out = spatial_transform_by_grid(im_t, points_grid, device)
    out_im = transforms.ToPILImage()(out[0])
    out_im.save("test.jpg")
    a = cv2.imread("test.jpg")
    b = cv2.imread("raw.jpg")
    print(((a - b) ** 2).mean())
    # im = cv2.imread("/home/wyq/DeepImageStitching-pytorch/align_data/train/input1/000001.jpg")
    # src = np.array([[0, 0], [127, 0], [0, 127], [127, 127]])
    # homograhy, _ = cv2.findHomography(src, src)
    # im_out = cv2.warpPerspective(im, homograhy, (128, 128))
    # print(np.abs(im_out - im).sum())

