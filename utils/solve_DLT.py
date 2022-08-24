from typing import Tuple
import torch.nn.functional as F
import torchgeometry
import torch.nn as nn
import torch

def solve_mesh_flow_DLT(mesh_flow: torch.Tensor, device: torch.device, patch_size: Tuple[int], image_size: Tuple[int]) -> Tuple[torch.Tensor]:
    # 假设像素为长度为1的正方形方块, mesh grid则为方块上的顶点, 以方块的中心点[0.5, 0.5]对其进行变换
    batch_size = mesh_flow.shape[0]
    unfold_mesh_flow = grid2pointgroup(mesh_flow, (2, 2))
    mesh_grid_X, mesh_grid_Y = torch.meshgrid(torch.arange(start=0, end=image_size[0] + 1, step=patch_size[0], dtype=torch.float, device=device),
        torch.arange(start=0, end=image_size[1] + 1, step=patch_size[1], dtype=torch.float, device=device,),
        indexing="xy",
    )
    mesh_grid = torch.cat([mesh_grid_X.unsqueeze(0), mesh_grid_Y.unsqueeze(0)], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)
    unfold_mesh_grid = grid2pointgroup(mesh_grid, (2, 2))
    solved_matrices = torchgeometry.get_perspective_transform(unfold_mesh_grid + unfold_mesh_flow, unfold_mesh_grid)

    # **Whether scale the homography matrices**
    # solved_matrices = H_scale(solved_matrices, patch_size[0], patch_size[1], unfold_mesh_grid.shape[0], device=device)

    # sort pixels into grid of mesh
    warped_mesh_grid = unfold_mesh_grid + unfold_mesh_flow
    points_grid = torch.meshgrid(torch.arange(start=0.5, end=image_size[0], step=1, dtype=torch.float, device=device), \
        torch.arange(start=0.5, end=image_size[1], step=1, dtype=torch.float, device=device), indexing='xy')
    points_grid = torch.cat([points_grid[0].unsqueeze(2), points_grid[1].unsqueeze(2)], dim=2).unsqueeze(0).expand(warped_mesh_grid.shape[0], -1, -1, -1)
    warped_mesh_grid = warped_mesh_grid.reshape(-1, 2, 2, 2)
    def GetCross(point1, point2, points_grid, mesh_grid):
        cross_product = (mesh_grid[:, point2[0]:point2[0] + 1, point2[1]:point2[1] + 1, 0] - mesh_grid[:, point1[0]:point1[0] + 1, point1[1]:point1[1] + 1, 0]) * \
            (points_grid[:, :, :, 1] - mesh_grid[:, point1[0]:point1[0] + 1, point1[1]:point1[1] + 1, 1]) - \
                (points_grid[:, :, :, 0] - mesh_grid[:, point1[0]:point1[0] + 1, point1[1]:point1[1] + 1, 0]) * \
                    (mesh_grid[:, point2[0]:point2[0] + 1, point2[1]:point2[1] + 1, 1] - mesh_grid[:, point1[0]:point1[0] + 1, point1[1]:point1[1] + 1, 1])
        return cross_product
    # cross1
    cross12 = GetCross((0, 0), (1, 0), points_grid, warped_mesh_grid)
    cross23 = GetCross((1, 0), (1, 1), points_grid, warped_mesh_grid)
    cross34 = GetCross((1, 1), (0, 1), points_grid, warped_mesh_grid)
    cross41 = GetCross((0, 1), (0, 0), points_grid, warped_mesh_grid)
    points_grid_mask = torch.bitwise_and((cross12 * cross34) >= 0, (cross23 * cross41) >= 0,).float()

    points_grid = points_grid.reshape(-1, image_size[0] * image_size[1], 2)
    # inf values
    ones = torch.ones_like(points_grid[:, :, 0:1])
    points_grid = torch.cat([points_grid, ones], dim=2)
    transformed_grid = torch.matmul(solved_matrices, points_grid.permute(0, 2, 1))
    x_s, y_s, t_s = torch.chunk(transformed_grid, 3, dim=1)
    # t_s = transformed_grid[:, 2, :]
    thresh = 1e-6
    t_s = torch.where((t_s < 0) & (t_s > - thresh), torch.full_like(t_s, -1e-6), t_s)
    t_s = torch.where((t_s >= 0) & (t_s < thresh), torch.full_like(t_s, 1e-6), t_s)
    x_coor = x_s / t_s
    y_coor = y_s / t_s

    warped_points_grid = torch.cat([x_coor, y_coor], dim=1).permute(0, 2, 1).reshape(batch_size, -1, image_size[0], image_size[1], 2)
    # warped_points_grid = torchgeometry.core.transform_points(solved_matrices, points_grid).reshape(batch_size, -1, image_size[0], image_size[1], 2)
    # print("debug here:", torch.where(torch.isnan(warped_points_grid)))
    points_grid_mask = points_grid_mask.reshape(batch_size, -1, image_size[0], image_size[1], 1)
    points_grid_weight = points_grid_mask.sum(dim=1)
    points_grid_weight[torch.where(points_grid_weight == 0)] = 1
    # print("debug here 2:", torch.where(points_grid_weight == 0))
    warped_points = (warped_points_grid * points_grid_mask).sum(dim=1) / points_grid_weight
    # print(points_grid_weight.shape, warped_points.shape)
    # if torch.numel(torch.where(torch.isnan(warped_points))[0]) != 0:
    #     print("debug here 1:", torch.where(torch.isnan(warped_points_grid)))
    #     print("debug here 2:", torch.where(torch.isnan(points_grid_mask)))
    #     print("debug here 3:", torch.where(torch.isnan(warped_points)))
    #     torch.save(warped_points_grid, '1.pt')
    #     torch.save(points_grid_mask, '2.pt')
    #     torch.save(points_grid_weight, '3.pt')
    #     print("debug here 4:", points_grid_weight[torch.where(torch.isnan(warped_points))[0], torch.where(torch.isnan(warped_points))[1],torch.where(torch.isnan(warped_points))[2]])
    warped_points = warped_points.permute(0, 3, 1, 2)

    return warped_points, solved_matrices

def solve_mesh_flow_DLT_triangle(mesh_flow:torch.Tensor, device:torch.device, patch_size:Tuple[int], image_size:Tuple[int], tris_index:torch.Tensor) -> Tuple[torch.Tensor]:
    # TODO: In future development, Delaunay mesh calculation should be completed outside the function
    batch_size = mesh_flow.shape[0]
    mesh_grid_X, mesh_grid_Y = torch.meshgrid(torch.arange(start=0, end=image_size[0] + 1, step=patch_size[0], dtype=torch.float, device=device),
        torch.arange(start=0, end=image_size[1] + 1, step=patch_size[1], dtype=torch.float, device=device,),
        indexing="xy",
    )
    mesh_grid = torch.cat([mesh_grid_X.unsqueeze(0), mesh_grid_Y.unsqueeze(0)], dim=0)
    mesh_grid_flat = mesh_grid.permute(1, 2, 0).reshape(-1, 2)
    origin_tris_samples = mesh_grid_flat[tris_index].unsqueeze(0).expand(batch_size, -1, -1, -1)
    origin_mesh_grid = torch.cat([mesh_grid_X.unsqueeze(0), mesh_grid_Y.unsqueeze(0)], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)
    warped_mesh_grid = (origin_mesh_grid + mesh_flow).permute(0, 2, 3, 1).reshape(batch_size, -1, 2)
    warped_tris_samples = warped_mesh_grid[:, tris_index, :]
    solved_matrices = solve_affine_DLT(warped_tris_samples.reshape(-1, 3, 2), origin_tris_samples.reshape(-1, 3, 2), device=device)
    # solved_matrices = H_scale(solved_matrices, patch_size[0], patch_size[1], solved_matrices.shape[0], device=device)

    points_grid = torch.meshgrid(torch.arange(start=0.5, end=image_size[0], step=1, dtype=torch.float, device=device), \
        torch.arange(start=0.5, end=image_size[1], step=1, dtype=torch.float, device=device), indexing='xy')
    points_grid = torch.cat([points_grid[0].unsqueeze(2), points_grid[1].unsqueeze(2)], dim=2).unsqueeze(0).expand(origin_tris_samples.reshape(-1, 3, 2).shape[0], -1, -1, -1)

    def GetCross(point1, point2, points_grid, mesh_grid):
        cross_product = (mesh_grid[:, 0:1, point2:point2 + 1, 0] - mesh_grid[:, 0:1, point1:point1 + 1, 0]) * (points_grid[:, :, :, 1] - mesh_grid[:, 0:1, point1:point1 + 1, 1]) - \
                (points_grid[:, :, :, 0] - mesh_grid[:, 0:1, point1:point1 + 1, 0]) * (mesh_grid[:, 0:1, point2:point2 + 1, 1] - mesh_grid[:, 0:1, point1:point1 + 1, 1])
        return cross_product

    warped_tris_samples = warped_tris_samples.reshape(-1, 1, 3, 2)
    cross12 = GetCross(0, 1, points_grid, warped_tris_samples)
    cross23 = GetCross(1, 2, points_grid, warped_tris_samples)
    cross31 = GetCross(2, 0, points_grid, warped_tris_samples)
    points_grid_mask = torch.bitwise_and((cross12 * cross23) >= 0, (cross23 * cross31) >= 0,).float()
    points_grid = points_grid.reshape(-1, image_size[0] * image_size[1], 2)
    # print(solved_matrices.shape, points_grid.shape)
    # warped_points_grid = torchgeometry.core.transform_points(solved_matrices, points_grid).reshape(batch_size, -1, image_size[0], image_size[1], 2)
    # rewrite points transformation using homography matrices
    ones = torch.ones_like(points_grid[:, :, 0:1])
    points_grid = torch.cat([points_grid, ones], dim=2)
    transformed_grid = torch.matmul(solved_matrices, points_grid.permute(0, 2, 1))
    x_s, y_s, t_s = torch.chunk(transformed_grid, 3, dim=1)
    x_s = x_s / t_s
    y_s = y_s / t_s
    warped_points_grid = torch.cat([x_s, y_s], dim=1).permute(0, 2, 1).reshape(batch_size, -1, image_size[0], image_size[1], 2)
    points_grid_mask = points_grid_mask.reshape(batch_size, -1, image_size[0], image_size[1], 1)
    points_grid_weight = points_grid_mask.sum(dim=1)
    points_grid_weight[torch.where(points_grid_weight == 0)] = 1
    warped_points = (warped_points_grid * points_grid_mask).sum(dim=1) / points_grid_weight
    # torch.save(points_grid_mask.sum(dim=1), "test.pt")
    warped_points = warped_points.permute(0, 3, 1, 2)

    return warped_points, solved_matrices

def solve_affine_DLT(points1:torch.Tensor, points2:torch.Tensor, device:torch.device) -> torch.Tensor:
    batch_size = points1.shape[0]
    A1 = torch.zeros((batch_size, 6, 6), dtype=torch.float, device=device)
    A1[:, 0:3, 0:2] = points1
    A1[:, 3:6, 3:5] = points1
    A1[:, 0:3, 2] = 1.
    A1[:, 3:6, 5] = 1.
    b = points2.permute(0, 2, 1).reshape(batch_size, -1)
    H_6 = torch.linalg.solve(A1, b)
    H_constant = torch.zeros(size=[batch_size, 3], device=device, dtype=torch.float)
    H_constant[:, 2] = 1.
    H_9 = torch.concat([H_6, H_constant], dim=1).reshape(batch_size, 3, 3)
    return H_9  

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
    batch_size, channels, height, width = img.shape
    grid = grid.reshape(batch_size, 2, -1) - 0.5
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
    base_y0 = base + y0 * width
    base_y1 = base + y1 * width
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    im_flat = img.permute(0, 2, 3, 1).reshape(-1, channels).float()
    Ia, Ib, Ic, Id = im_flat[idx_a], im_flat[idx_b], im_flat[idx_c], im_flat[idx_d]

    x0_f = x0.float()
    x1_f = x1.float()
    y0_f = y0.float()
    y1_f = y1.float()
    
    wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
    wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
    wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
    wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)


    output = wa * Ia + wb * Ib + wc * Ic + wd * Id
    output = output.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)

    return output 


# Test
if __name__ == "__main__":
    from PIL import Image
    from torchvision import transforms
    from scipy.spatial import Delaunay
    import sys
    sys.path.append("..")
    from models.mesh_flow_upsample import mesh_flow_upsampling
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0")
    mesh_grid_X, mesh_grid_Y = torch.meshgrid(torch.arange(9, dtype=torch.float, device=device),
        torch.arange(9, dtype=torch.float, device=device),
        indexing="xy",
    )
    mesh_grid = torch.cat([mesh_grid_X.unsqueeze(0), mesh_grid_Y.unsqueeze(0)], dim=0)
    mesh_grid_flat = mesh_grid.permute(1, 2, 0).reshape(-1, 2)
    tris = Delaunay(mesh_grid_flat.cpu().numpy())
    tris_samples_indices = torch.from_numpy(tris.simplices).type(torch.int64)
    # mesh_flow = torch.randn(8, 2, 17, 17)
    # mesh_flow = torch.tensor([(1, 1), (1, 1), (10, 1), (10, 1)], dtype=torch.float).permute(1, 0).reshape(2, 2, 2).unsqueeze(0)
    mesh_flow = torch.tensor([(0, 0), (0, 100), (0, 0), (0, 0), (-100, -100), (0, 0), (0, 0), (0, 0), (0, 0)], dtype=torch.float, device=device).permute(1, 0).reshape(2, 3, 3).unsqueeze(0)
    mesh_flow = mesh_flow_upsampling(mesh_flow, (3, 3), (9, 9), (1280, 1280), 1, device)
    mesh_flow = torch.randn((1, 2, 9, 9)) * 10
    mesh_flow = mesh_flow.to(device)
    # print(mesh_flow)
    # exit()
    # mesh_flow = torch.tensor([(0, 0), (0, 0), (-5, 1), (5, 1)], dtype=torch.float).permute(1, 0).reshape(2, 2, 2).unsqueeze(0)

    b1, c1 = solve_mesh_flow_DLT_triangle(mesh_flow, device, (50, 50), (400, 400), tris_index=tris_samples_indices)
    b, c = solve_mesh_flow_DLT(mesh_flow, device, (50, 50), (400, 400))
    # cv_im = cv2.imread("../im_test/square.jpg")
    # b_x = b[0, 0, :, :].numpy()
    # b_y = b[0, 1, :, :].numpy()
    # out = cv2.remap(cv_im, b_x, b_y, cv2.INTER_LINEAR)
    # cv2.imwrite("../im_test/warp_square_remap.jpg", out)
    # a = solve_mesh_flow_DLT_triangle(mesh_flow, torch.device("cpu"), (64, 64), (128, 128))
    # print(b)
    img = Image.open("../im_test/square_xs.jpg").convert("RGB")
    input_tensor = transforms.ToTensor()(img).unsqueeze(0)
    input_tensor = input_tensor.to(device=device)
    warp_tensor = spatial_transform_by_grid(input_tensor, b, device)
    warp_img = transforms.ToPILImage()(warp_tensor.squeeze(0))
    warp_img.save("../im_test/warp_square_xs.jpg")
    warp_tensor2 = spatial_transform_by_grid(input_tensor, b1, device)
    warp_img = transforms.ToPILImage()(warp_tensor2.squeeze(0))
    warp_img.save("../im_test/warp_square_xs2.jpg")
    # print(torch.where(warp_mask < 0))
    # a = torch.arange(16).reshape(2, 2, 2, 2)
    # b = a + 1
    # ab = torch.cat([a.unsqueeze(1), b.unsqueeze(1)], dim=1)
    # c = torch.randn(size=(2, 1, 2, ))

