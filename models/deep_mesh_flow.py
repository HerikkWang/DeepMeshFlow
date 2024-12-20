import torch.nn as nn
import torch
import torch.nn.functional as F
import argparse
from scipy.spatial import Delaunay

import sys
sys.path.append("..")

from models.model_parts import feature_extractor, mask_predictor, mesh_estimator_body, mesh_estimator_head, mesh_selector
from utils.solve_DLT import solve_mesh_flow_DLT, spatial_transform_by_grid, solve_mesh_flow_DLT_triangle

class DeepMeshFlow(nn.Module):
    def __init__(self, args:argparse.ArgumentParser, device:torch.device, stage:str="train") -> None:
        super().__init__()
        self.args = args
        self.stage = stage
        self.img_size = args.image_size
        self.patch_size = (args.image_size[0] // (args.mesh_size_3[0] - 1), args.image_size[1] // (args.mesh_size_3[1] - 1))
        self.device = device
        self.feature_extractor = feature_extractor(in_channels=1, out_channels=1)
        self.mask_predictor = mask_predictor(in_channels=1)
        self.estimator_body = mesh_estimator_body
        self.estimator_head_0 = mesh_estimator_head(args.mesh_size_1, args.mesh_size_3, self.img_size, device=self.device)
        self.estimator_head_1 = mesh_estimator_head(args.mesh_size_2, args.mesh_size_3, self.img_size, device=self.device)
        self.estimator_head_2 = mesh_estimator_head(args.mesh_size_3, args.mesh_size_3, self.img_size, device=self.device)
        self.mesh_selector = mesh_selector(in_channels=2, out_shape=(3, 17, 17))
        # calculate Delaunay Triangulation
        mesh_grid_X, mesh_grid_Y = torch.meshgrid(torch.arange(args.mesh_size_3[0], dtype=torch.float, device=device),
            torch.arange(args.mesh_size_3[1], dtype=torch.float, device=device,),
            indexing="xy",
        )
        mesh_grid = torch.cat([mesh_grid_X.unsqueeze(0), mesh_grid_Y.unsqueeze(0)], dim=0)
        mesh_grid_flat = mesh_grid.permute(1, 2, 0).reshape(-1, 2)
        tris = Delaunay(mesh_grid_flat.cpu().numpy())
        self.tris_samples_indices = torch.from_numpy(tris.simplices).type(torch.int64)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_normal_(m.weight)
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        input_orig1 = x[:, 0:3, :, :].mean(dim=1, keepdim=True)
        input_orig2 = x[:, 3:6, :, :].mean(dim=1, keepdim=True)
        input1 = x[:, 6:9, :, :].mean(dim=1, keepdim=True)
        input2 = x[:, 9:12, :, :].mean(dim=1, keepdim=True)
        feature1 = self.feature_extractor(input1)
        feature2 = self.feature_extractor(input2)
        mask1 = self.mask_predictor(input1)
        mask2 = self.mask_predictor(input2)
        feature_processed = torch.cat([feature1 * mask1, feature2 * mask2], dim=1)
        out = self.estimator_body(feature_processed)

        mesh_flow_0 = self.estimator_head_0(out).unsqueeze(1)
        mesh_flow_1 = self.estimator_head_1(out).unsqueeze(1)
        mesh_flow_2 = self.estimator_head_2(out).unsqueeze(1)
        mesh_flow = torch.cat([mesh_flow_0, mesh_flow_1, mesh_flow_2], dim=1)
        mesh_index = torch.argmax(self.mesh_selector(torch.cat([input1, input2], dim=1)), dim=1, keepdim=True)
        mesh_index = mesh_index.unsqueeze(2).expand(-1, -1, 2, -1, -1)
        mesh_out = torch.gather(mesh_flow, dim=1, index=mesh_index).squeeze(1)
        mesh_out = mesh_flow_2.squeeze(1)

        feature_processed_inv = torch.cat([feature2 * mask2, feature1 * mask1], dim=1)
        out_inv = self.estimator_body(feature_processed_inv)
        mesh_flow_0_inv = self.estimator_head_0(out_inv).unsqueeze(1)
        mesh_flow_1_inv = self.estimator_head_1(out_inv).unsqueeze(1)
        mesh_flow_2_inv = self.estimator_head_2(out_inv).unsqueeze(1)
        mesh_flow_inv = torch.cat([mesh_flow_0_inv, mesh_flow_1_inv, mesh_flow_2_inv], dim=1)
        mesh_index_inv = torch.argmax(self.mesh_selector(torch.cat([input2, input1], dim=1)), dim=1, keepdim=True)
        mesh_index_inv = mesh_index_inv.unsqueeze(2).expand(-1, -1, 2, -1, -1)
        mesh_out_inv = torch.gather(mesh_flow_inv, dim=1, index=mesh_index_inv).squeeze(1)
        
        # use mesh warp original images
        # warped_grid, homography_grid = solve_mesh_flow_DLT_triangle(mesh_flow=mesh_out, device=self.device, image_size=self.img_size, patch_size=self.patch_size, tris_index=self.tris_samples_indices)
        # warped_grid_inv, homography_grid_inv = solve_mesh_flow_DLT_triangle(mesh_flow=mesh_out_inv, device=self.device, image_size=self.img_size, patch_size=self.patch_size, tris_index=self.tris_samples_indices)
        warped_grid, homography_grid = solve_mesh_flow_DLT(mesh_flow=mesh_out, device=self.device, image_size=self.img_size, patch_size=self.patch_size)
        warped_grid_inv, homography_grid_inv = solve_mesh_flow_DLT(mesh_flow=mesh_out_inv, device=self.device, image_size=self.img_size, patch_size=self.patch_size)
        im1_warp = spatial_transform_by_grid(input_orig1, warped_grid, device=self.device)
        im2_warp = spatial_transform_by_grid(input_orig2, warped_grid_inv, device=self.device)
        mask_ones = torch.ones_like(x[:, 0:3, :, :])
        ones_mask1_warp = spatial_transform_by_grid(mask_ones, warped_grid, device=self.device)
        ones_mask2_warp = spatial_transform_by_grid(mask_ones, warped_grid_inv, device=self.device)
        ones_mask1_warp[torch.where(ones_mask1_warp < 0.3)] = 0
        ones_mask2_warp[torch.where(ones_mask2_warp < 0.3)] = 0
        feature1_warp = self.feature_extractor(im1_warp)
        feature2_warp = self.feature_extractor(im2_warp)
        feature1_orig = self.feature_extractor(input_orig1)
        feature2_orig = self.feature_extractor(input_orig2)
        mask1_orig = self.mask_predictor(input_orig1)
        mask2_orig = self.mask_predictor(input_orig2)
        mask1_warp = spatial_transform_by_grid(mask1_orig, warped_grid, device=self.device)
        mask2_warp = spatial_transform_by_grid(mask2_orig, warped_grid_inv, device=self.device)

        return feature1_warp, feature2_warp, feature1_orig, feature2_orig, mask1_orig, mask2_orig, mask1_warp, mask2_warp, \
            homography_grid, homography_grid_inv, warped_grid, warped_grid_inv, mesh_out, mesh_out_inv, im1_warp, im2_warp, ones_mask1_warp, ones_mask2_warp

if __name__ == "__main__":
    from torchsummary import summary
    from config.option import args
    from dataset import align_dataset
    from torch.utils.data import DataLoader
    from torchvision import transforms
    train_dataset = align_dataset(args=args)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)
    device = torch.device("cuda:0")
    model = DeepMeshFlow(args, device=device)
    model.to(device)
    for i, sample_batch in enumerate(train_loader):
        input_tensor = sample_batch[0].to(device)
        img_names = sample_batch[1]
        raw_img1 = input_tensor[:, 0:3, :, :]
        raw_img2 = input_tensor[:, 3:6, :, :]
        # loss for DeepMeshFlow
        if args.model == "DeepMeshFlow":
            feature1_warp, feature2_warp, feature1_orig, feature2_orig, mask1_orig, mask2_orig, mask1_warp, mask2_warp, \
            homography_grid, homography_grid_inv, warped_grid, warped_grid_inv, mesh_out, mesh_out_inv, im1_warp, im2_warp, ones_mask1_warp, ones_mask2_warp = model(input_tensor)
        im1 = transforms.ToPILImage()(raw_img1[0])
        im2 = transforms.ToPILImage()(raw_img2[0])
        raw_im1_warp = spatial_transform_by_grid(raw_img1, warped_grid, device=device)
        raw_im2_warp = spatial_transform_by_grid(raw_img2, warped_grid_inv, device=device)
        raw_im1_warp = transforms.ToPILImage()(raw_im1_warp[0])
        raw_im2_warp = transforms.ToPILImage()(raw_im2_warp[0])
        im1.save("im1.jpg")
        im2.save("im2.jpg")
        raw_im1_warp.save("im1_warp.jpg")
        raw_im2_warp.save("im2_warp.jpg")
        break
    # print(summary(model, (12, 128, 128), device='cpu'))