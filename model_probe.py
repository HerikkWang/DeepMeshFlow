import torch
from models.deep_mesh_flow import DeepMeshFlow
from config.option import args
from PIL import Image
from dataset import align_dataset
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import cv2
import numpy as np

# device = torch.device("cuda:0")
# model = DeepMeshFlow(args, device=device)
# model.load_state_dict(torch.load("nan.pth")['state_dict'])
# model = model.to(device)
# transform_jitter = transforms.Compose([
#             transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2),
#             transforms.ToTensor(),
#         ])
# train_dataset = align_dataset(args=args)
# subset = Subset(train_dataset, list(range(19824, 19824 + 8)))
# # print(subset[0])
# print(len(subset))
# train_loader = DataLoader(dataset=subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
# im1 = Image.open("/home/wyq/DeepImageStitching-pytorch/align_data/train/input1/000001.jpg")
# im1.save("im_test/raw.jpg")
# im2 = Image.open("/home/wyq/DeepImageStitching-pytorch/align_data/train/input2/019825.jpg")
# im1_aug = transform_jitter(im1).unsqueeze(0)
# im2_aug = transform_jitter(im2).unsqueeze(0)
# im1 = transforms.ToTensor()(im1).unsqueeze(0)
# im2 = transforms.ToTensor()(im2).unsqueeze(0)
# input_tensor = torch.cat([im1, im2, im1_aug, im2_aug], dim=1)
# input_tensor = torch.load("input_tensor.pt").to(device)
# mask1_orig = torch.load("mask1_orig.pt")
# mask2_warp = torch.load("mask2_warp.pt")
# feature1_orig = torch.load("feature1_orig.pt")
# feature2_warp = torch.load("feature2_warp.pt")
# if torch.any(torch.isnan(mask1_orig)):
#     print("debug1")
# if torch.any(torch.isnan(mask2_warp)):
#     print("debug2")
# if torch.any(torch.isnan(feature1_orig)):
#     print("debug3")
# if torch.any(torch.isnan(feature2_warp)):
#     print("debug4")

# for i in range(10):
#     im_aug = transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2)(im1)
#     im_aug.save("im_test/aug{num}.jpg".format(num=i))


# with torch.no_grad():
    # for i, sample_batch in enumerate(train_loader):
    #     input_tensor = sample_batch[0]
    #     input_tensor = input_tensor.to(device)
    # for name in model.feature_extractor.state_dict():
    #     print(name)
    # print(model.feature_extractor.state_dict()['extractor.0.weight'])
# model.train()
# feature1_warp, feature2_warp, feature1_orig, feature2_orig, mask1_orig, mask2_orig, mask1_warp, mask2_warp, \
#     homography_grid, homography_grid_inv, warped_grid, warped_grid_inv = model(input_tensor)

# ln = torch.sum(mask1_warp * mask2_orig * torch.abs(feature2_orig - feature1_warp)) / torch.sum(mask1_warp * mask2_orig)

# ln_inv = torch.sum(mask1_orig * mask2_warp * torch.abs(feature1_orig - feature2_warp)) / torch.sum(mask1_orig * mask2_warp)
# # TODO: Does L ab loss should be normalized ?
# L_ab = -1 * torch.abs(feature1_orig - feature2_orig).mean()
# Identity = torch.eye(3, dtype=torch.float, device=device).unsqueeze(0).expand(homography_grid.shape[0], -1, -1)
# L_inverse = torch.mean(torch.abs(torch.matmul(homography_grid, homography_grid_inv) - Identity))
# loss = ln + ln_inv + args.loss_weight_lambda * L_ab + args.loss_weight_mu * L_inverse
# if torch.isnan(ln):
#     print("ln is nan.")
# if torch.isnan(ln_inv):
#     print("ln_inv is nan.")
# if torch.isnan(L_ab):
#     print("L_ab is nan.")
# if torch.isnan(L_inverse):
#     print("L_inverse is nan.")

    # print(torch.isnan(feature1_orig))

point_src1 = np.float32([[0, 0], [64, 0], [64, 64], [0, 64]])
point_dst1 = np.float32([[0, 0], [64, 0], [64, 94], [0, 64]])
point_src2 = np.float32([[0, 64], [64, 64], [0, 128], [64, 128]])
point_dst2 = np.float32([[0, 64], [64, 94], [0, 128], [64, 128]])
H1 = cv2.getPerspectiveTransform(point_dst1, point_src1)
H2 = cv2.getPerspectiveTransform(point_dst2, point_src2)
test_point = np.array([32, 79, 1], dtype=np.float32)
x1 = (test_point * H1[0]).sum() / (test_point * H1[2]).sum()
y1 = (test_point * H1[1]).sum() / (test_point * H1[2]).sum()
x2 = (test_point * H2[0]).sum() / (test_point * H2[2]).sum()
y2 = (test_point * H2[1]).sum() / (test_point * H2[2]).sum()
print(x1, y1, x2, y2)
print(H1)