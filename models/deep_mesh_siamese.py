import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
import argparse
from scipy.spatial import Delaunay

import sys
sys.path.append("..")

from models.model_parts import feature_extractor, mask_predictor, mesh_estimator_body, mesh_estimator_head, mesh_selector, resnet34
from utils.solve_DLT import solve_mesh_flow_DLT, solve_mesh_flow_DLT_triangle, spatial_transform_by_grid

class SiameseDeepMesh(torch.nn.Module):
    def __init__(self, args:argparse.ArgumentParser, device:torch.device, stage:str="train") -> None:
        super(SiameseDeepMesh, self).__init__()
        self.args = args
        self.stage = stage
        self.img_size = args.image_size
        self.patch_size = (args.image_size[0] // (args.mesh_size_3[0] - 1), args.image_size[1] // (args.mesh_size_3[1] - 1))
        self.device = device
        self.feature_extractor = torch.nn.Sequential(*list(torchvision.models.resnet18(pretrained=True).children())[:-2])
        self.estimator_head_0 = mesh_estimator_head(args.mesh_size_1, args.mesh_size_3, self.img_size, device=self.device, in_channels=1024)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1))
    
    def forward(self, x):
        input_orig1 = x[:, 0:3, :, :]
        input_orig2 = x[:, 3:6, :, :]
        input1 = x[:, 6:9, :, :]
        input2 = x[:, 9:12, :, :]
        feature1 = self.feature_extractor(input1)
        feature2 = self.feature_extractor(input2)
        features = torch.cat([feature1, feature2], dim=1)
        features = self.avgpool(features).squeeze(-1).squeeze(-1)
        mesh_flow_out = self.estimator_head_0(features)
        features_inv = torch.cat([feature2, feature1], dim=1)
        features_inv = self.avgpool(features_inv).squeeze(-1).squeeze(-1)
        mesh_flow_out_inv = self.estimator_head_0(features_inv)

        warped_grid, homography_grid = solve_mesh_flow_DLT(mesh_flow=mesh_flow_out, device=self.device, image_size=self.img_size, patch_size=self.patch_size)
        warped_grid_inv, homography_grid_inv = solve_mesh_flow_DLT(mesh_flow=mesh_flow_out_inv, device=self.device, image_size=self.img_size, patch_size=self.patch_size)
        im1_warp = spatial_transform_by_grid(input_orig1, warped_grid, device=self.device)
        im2_warp = spatial_transform_by_grid(input_orig2, warped_grid_inv, device=self.device)
        mask_ones = torch.ones_like(input_orig1)
        mask1_warp = spatial_transform_by_grid(mask_ones, warped_grid, device=self.device)
        mask2_warp = spatial_transform_by_grid(mask_ones, warped_grid_inv, device=self.device)
        mask1_warp[torch.where(mask1_warp < 0.3)] = 0
        mask2_warp[torch.where(mask2_warp < 0.3)] = 0
        

        return im1_warp, im2_warp, homography_grid, homography_grid_inv, mask1_warp, mask2_warp

if __name__ == "__main__":
    from torchsummary import summary
    from config.option import args
    from dataset import align_dataset
    from torch.utils.data import DataLoader
    from torchvision import transforms
    # train_dataset = align_dataset(args=args)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)
    device = torch.device("cpu")
    model = SiameseDeepMesh(args, device=device)
    # model.to(device)
    # for i, sample_batch in enumerate(train_loader):
    #     input_tensor = sample_batch[0].to(device)
    #     img_names = sample_batch[1]
    #     raw_img1 = input_tensor[:, 0:3, :, :]
    #     raw_img2 = input_tensor[:, 3:6, :, :]
    #     # loss for DeepMeshFlow
    #     if args.model == "DeepMeshFlow":
    #         feature1_warp, feature2_warp, feature1_orig, feature2_orig, mask1_orig, mask2_orig, mask1_warp, mask2_warp, \
    #         homography_grid, homography_grid_inv, warped_grid, warped_grid_inv, mesh_out, mesh_out_inv, im1_warp, im2_warp, ones_mask1_warp, ones_mask2_warp = model(input_tensor)
    #     im1 = transforms.ToPILImage()(raw_img1[0])
    #     im2 = transforms.ToPILImage()(raw_img2[0])
    #     raw_im1_warp = spatial_transform_by_grid(raw_img1, warped_grid, device=device)
    #     raw_im2_warp = spatial_transform_by_grid(raw_img2, warped_grid_inv, device=device)
    #     raw_im1_warp = transforms.ToPILImage()(raw_im1_warp[0])
    #     raw_im2_warp = transforms.ToPILImage()(raw_im2_warp[0])
    #     im1.save("im1.jpg")
    #     im2.save("im2.jpg")
    #     raw_im1_warp.save("im1_warp.jpg")
    #     raw_im2_warp.save("im2_warp.jpg")
    #     break
    print(summary(model, (12, 128, 128), device='cpu'))