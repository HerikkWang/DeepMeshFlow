import torch
from models.deep_mesh_flow import DeepMeshFlow
from config.option import args
from PIL import Image
from dataset import align_dataset
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

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
im1 = Image.open("/home/wyq/DeepImageStitching-pytorch/align_data/train/input1/000001.jpg")
im1.save("im_test/raw.jpg")
# im2 = Image.open("/home/wyq/DeepImageStitching-pytorch/align_data/train/input2/019825.jpg")
# im1_aug = transform_jitter(im1).unsqueeze(0)
# im2_aug = transform_jitter(im2).unsqueeze(0)
# im1 = transforms.ToTensor()(im1).unsqueeze(0)
# im2 = transforms.ToTensor()(im2).unsqueeze(0)
# input_tensor = torch.cat([im1, im2, im1_aug, im2_aug], dim=1)
# input_tensor = input_tensor.to(device=device)

for i in range(10):
    im_aug = transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2)(im1)
    im_aug.save("im_test/aug{num}.jpg".format(num=i))


# with torch.no_grad():
#     for i, sample_batch in enumerate(train_loader):
#         input_tensor = sample_batch[0]
#         input_tensor = input_tensor.to(device)
#     # for name in model.feature_extractor.state_dict():
#     #     print(name)
#     # print(model.feature_extractor.state_dict()['extractor.0.weight'])
#         feature1_warp, feature2_warp, feature1_orig, feature2_orig, mask1_orig, mask2_orig, mask1_warp, mask2_warp, \
#             homography_grid, homography_grid_inv, warped_grid, warped_grid_inv = model(input_tensor)

#         ln = torch.sum(mask1_warp * mask2_orig * torch.abs(feature2_orig - feature1_warp)) / torch.sum(mask1_warp * mask2_orig)

#         ln_inv = torch.sum(mask1_orig * mask2_warp * torch.abs(feature1_orig - feature2_warp)) / torch.sum(mask1_orig * mask2_warp)
#         # TODO: Does L ab loss should be normalized ?
#         L_ab = -1 * torch.abs(feature1_orig - feature2_orig).mean()
#         Identity = torch.eye(3, dtype=torch.float, device=device).unsqueeze(0).expand(homography_grid.shape[0], -1, -1)
#         L_inverse = torch.mean(torch.abs(torch.matmul(homography_grid, homography_grid_inv) - Identity))
#         loss = ln + ln_inv + args.loss_weight_lambda * L_ab + args.loss_weight_mu * L_inverse
#         if torch.isnan(ln):
#             print("ln is nan.")
#         if torch.isnan(ln_inv):
#             print("ln_inv is nan.")
#         if torch.isnan(L_ab):
#             print("L_ab is nan.")
#         if torch.isnan(L_inverse):
#             print("L_inverse is nan.")

    # print(torch.isnan(feature1_orig))
