import torch.nn as nn
import torch
import argparse
from model_parts import *

class DeepMeshFlow(nn.Module):
    def __init__(self, args:argparse.ArgumentParser) -> None:
        super().__init__()
        self.img_size = args.image_size
        self.feature_extractor = feature_extractor(in_channels=1, out_channels=1)
        self.mask_predictor = mask_predictor(in_channels=1)
        self.estimator_body = mesh_estimator_body
        self.estimator_head_0 = mesh_estimator_head(args.mesh_size_1, args.mesh_size_3, self.img_size)
        self.estimator_head_1 = mesh_estimator_head(args.mesh_size_2, args.mesh_size_3, self.img_size)
        self.estimator_head_2 = mesh_estimator_head(args.mesh_size_3, args.mesh_size_3, self.img_size)
        self.mesh_selector = mesh_selector(2, (3, 17, 17))
    
    def forward(self, x):
        input1 = x[:, 0:1, :, :]
        input2 = x[:, 1:2, :, :]
        feature1 = self.feature_extractor(input1)
        feature2 = self.feature_extractor(input2)
        mask1 = self.feature_extractor(input1)
        mask2 = self.feature_extractor(input2)
        feature_processed = torch.cat([feature1 * mask1, feature2 * mask2], dim=1)
        out = self.estimator_body(feature_processed)

        mesh_flow_0 = self.estimator_head_0(out)
        mesh_flow_1 = self.estimator_head_1(out)
        mesh_flow_2 = self.estimator_head_2(out)
        mesh_flow = torch.cat([mesh_flow_0, mesh_flow_1, mesh_flow_2], dim=1)
        mesh_index = torch.argmax(self.mesh_selector(x), dim=1, keepdim=True)
        mesh_out = torch.gather(mesh_flow, dim=1, index=mesh_index)
        return mesh_out, mask1, mask2, feature1, feature2

if __name__ == "__main__":
    from torchsummary import summary
    model = DeepMeshFlow()
    print(summary(model, (2, 128, 128), device='cpu'))