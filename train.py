import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import align_dataset

from config.option import args
from utils.get_logger import Get_logger
from utils.metrics import calc_psnr
from utils.solve_DLT import solve_mesh_flow_DLT, spatial_transform_by_grid
from utils.toolkit import tensor2img
from pytorch_msssim.ssim import ssim
from models.deep_mesh_flow import DeepMeshFlow

import os

def train(args):
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # initialize logging file
    logger = Get_logger(os.path.join("log", args.exp_name + ".log"))
    logger.info("Parameters Setting:")
    for i in args.__dict__:
        logger.info("{param_name}: {param_value}".format(param_name=i, param_value=args.__dict__[i]))

    model = DeepMeshFlow(args=args, device=device)
    model = model.to(device)
    train_dataset = align_dataset(args=args)
    val_dataset = align_dataset(args=args, validation=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=args.weight_decay)

    # result save
    writer = SummaryWriter(os.path.join("runs", args.exp_name))
    model_save_path = os.path.join("checkpoints", args.exp_name)
    if not os.path.isdir(model_save_path):
        os.mkdir(model_save_path)
    
    print_every_iter = args.print_every_iter
    record_every_iter = args.record_every_iter
    save_every_epoch = args.save_every_epoch
    
    epochs = args.epoch
    start_epoch = 0
    tb_index = 0 + ((len(train_dataset) // args.batch_size) // args.record_every_iter) * start_epoch
    patch_size = (args.image_size[0] // (args.mesh_size_3[0] - 1), args.image_size[1] // (args.mesh_size_3[1] - 1))
    logger.info("Training start.")
    if args.resume:
        loaded = torch.load(os.path.join("checkpoints", args.exp_name, args.model_name + "_latest.pth"))
        model.load_state_dict(loaded['state_dict'])
        optimizer = torch.optim.Adam(model.parameters(), lr=loaded['lr'])
        start_epoch = loaded['epoch']
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = [0., 0., 0., 0., 0.]
        print_loss = 0.
        for i, sample_batch in enumerate(train_loader):
            input_tensor = sample_batch[0].to(device)
            # image_name = sample_batch[1].to(device)
            feature1_warp, feature2_warp, feature1_orig, feature2_orig, mask1_orig, mask2_orig, mask1_warp, mask2_warp, \
                homography_grid, homography_grid_inv, _, _ = model(input_tensor)

            ln = torch.sum(mask1_warp * mask2_orig * torch.abs(feature1_warp - feature2_orig)) / torch.sum(mask1_warp * mask2_orig)
            ln_inv = torch.sum(mask1_orig * mask2_warp * torch.abs(feature1_orig - feature2_warp)) / torch.sum(mask1_orig * mask2_warp)
            # TODO: Does L ab loss should be normalized ?
            L_ab = -1 * torch.abs(feature1_orig - feature2_orig).mean()
            Identity = torch.eye(3, dtype=torch.float, device=device).unsqueeze(0).expand(homography_grid.shape[0], -1, -1)
            L_inverse = torch.mean(torch.abs(torch.matmul(homography_grid, homography_grid_inv) - Identity))
            loss = ln + ln_inv + args.loss_weight_lambda * L_ab + args.loss_weight_mu * L_inverse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss[0] += loss.item()
            running_loss[1] += ln.item()
            running_loss[2] += ln_inv.item()
            running_loss[3] += L_ab.item()
            running_loss[4] += L_inverse.item()
            print_loss += loss.item()
            
            if (i + 1) % print_every_iter == 0:
                logger.info('[{}, {}] running_loss = {:.5f}, learning_rate = {:.7f}'.format(epoch + 1, i + 1, print_loss / print_every_iter, optimizer.state_dict()['param_groups'][0]['lr']))                
                print_loss = 0.
            if (i + 1) % record_every_iter == 0:
                writer.add_scalar("loss/Ln_ab", running_loss[1] / record_every_iter, tb_index)
                writer.add_scalar("loss/Ln_ba", running_loss[2] / record_every_iter, tb_index)
                writer.add_scalar("loss/L", running_loss[3] / record_every_iter, tb_index)
                writer.add_scalar("loss/L_inversable", running_loss[4] / record_every_iter, tb_index)
                writer.add_scalar("loss/loss_all", running_loss[0] / record_every_iter, tb_index)
                tb_index += 1
                for j in range(len(running_loss)):
                    running_loss[j] = 0.
        scheduler.step()

        # model save
        if (epoch + 1) % save_every_epoch == 0:
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'lr':optimizer.state_dict()['param_groups'][0]['lr'],
            }, os.path.join(model_save_path, "align{epoch}.pth".format(epoch=epoch + 1)))
        # save latest model
        torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'lr':optimizer.state_dict()['param_groups'][0]['lr'],
            }, os.path.join(model_save_path, "align_latest.pth"))
        
        # model validation
        model.eval()
        ssim_result1 = []
        ssim_result2 = []
        psnr_result1 = []
        psnr_result2 = []
        with torch.no_grad():
            for i, sample_batch in enumerate(val_loader):
                input_tensor = sample_batch[0].to(device)
                # input_tensor_aug = sample_batch[1].to(device)
                feature1_warp, feature2_warp, feature1_orig, feature2_orig, mask1_orig, mask2_orig, mask1_warp, mask2_warp, \
                    homography_grid, homography_grid_inv, warped_grid, warped_grid_inv = model(input_tensor)
                img1 = input_tensor[:, 0:3, :, :]
                img2 = input_tensor[:, 3:6, :, :]
                ones_mask = torch.ones_like(img1, dtype=torch.float, device=device)
                img1_warp = spatial_transform_by_grid(img1, warped_grid, device=device)
                img2_warp = spatial_transform_by_grid(img2, warped_grid_inv, device=device)
                img1_mask_warp = spatial_transform_by_grid(ones_mask, warped_grid, device=device)
                img2_mask_warp = spatial_transform_by_grid(ones_mask, warped_grid, device=device)
                img1_mask_warp[torch.where(img1_mask_warp < 0.4)] = 0
                img2_mask_warp[torch.where(img2_mask_warp < 0.4)] = 0
                img1_mask_warp = img1_mask_warp.bool()
                img2_mask_warp = img2_mask_warp.bool()

                img1 = tensor2img(img1)
                img2 = tensor2img(img2)
                img1_warp = tensor2img(img1_warp)
                img2_warp = tensor2img(img2_warp)
                if i == 0:
                    writer.add_image("val/image1",img1[0, ...], epoch)
                    writer.add_image("val/image2",img2[0, ...], epoch)
                    writer.add_image("val/image1 warp",img1_warp[0, ...], epoch)
                    writer.add_image("val/image2 warp",img2_warp[0, ...], epoch)
                img1 = img1.float()
                img2 = img2.float()
                img1_warp = img1_warp.float()
                img2_warp = img2_warp.float()
                ssim_tensor1 = ssim(img1, img2_warp * img2_mask_warp, data_range=255, size_average=False)
                ssim_tensor2 = ssim(img2, img1_warp * img1_mask_warp, data_range=255, size_average=False)
                psnr_tensor1 = calc_psnr(img1, img2_warp * img2_mask_warp)
                psnr_tensor2 = calc_psnr(img2, img1_warp * img1_mask_warp)
                ssim_result1.append(ssim_tensor1)
                ssim_result2.append(ssim_tensor2)
                psnr_result1.append(psnr_tensor1)
                psnr_result2.append(psnr_tensor2)
        psnr_result1 = torch.cat(psnr_result1, dim=0)
        psnr_result2 = torch.cat(psnr_result2, dim=0)
        # avoid infinite psnr value result
        psnr_result1 = psnr_result1[~torch.isinf(psnr_result1)]
        psnr_result2 = psnr_result2[~torch.isinf(psnr_result2)]
        psnr_value1 = torch.mean(psnr_result1)
        psnr_value2 = torch.mean(psnr_result2)
        ssim_value1 = torch.mean(torch.cat(ssim_result1), dim=0)
        ssim_value2 = torch.mean(torch.cat(ssim_result2), dim=0)
        logger.info("Epoch {epoch} | psnr 1 {psnr_value1} | ssim 1 {ssim_value1} | psnr 2 {psnr_value2} | ssim 2 {ssim_value2} |". \
            format(epoch=epoch + 1, psnr_value1=psnr_value1, ssim_value1=ssim_value1, psnr_value2=psnr_value2, ssim_value2=ssim_value2))
        writer.add_scalar("metrics/psnr1", psnr_value1, epoch)
        writer.add_scalar("metrics/psnr2", psnr_value2, epoch)
        writer.add_scalar("metrics/ssim1", ssim_value1, epoch)
        writer.add_scalar("metrics/ssim2", ssim_value2, epoch)

    logger.info("training complete.")

if __name__ == "__main__":
    train(args=args)





