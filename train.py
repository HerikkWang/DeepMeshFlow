import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import align_dataset

from config.option import args
from utils.get_logger import Get_logger
from utils.metrics import calc_psnr
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

    model = DeepMeshFlow(args=args)
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
    logger.info("Training start.")
    if args.resume:
        loaded = torch.load(os.path.join("checkpoints", args.exp_name, args.model_name + "_latest.pth"))
        model.load_state_dict(loaded['state_dict'])
        optimizer = torch.optim.Adam(model.parameters(), lr=loaded['lr'])
        start_epoch = loaded['epoch']
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = [0., 0., 0., 0.]
        print_loss = 0.
        for i, sample_batch in enumerate(train_loader):
            patch_size = (args.image_size[0] // (args.mesh_size_3[0] - 1), args.image_size[1] // (args.mesh_size_3[1] - 1))
            input_tensor = sample_batch[0].to(device)
            input_tensor_aug = sample_batch[1].to(device)
            mesh_flow, mask1, mask2, feature1, feature2 = model(input_tensor_aug)
            mesh_unfolder = nn.Unfold(kernel_size=(2, 2))
            unfold_mesh_flow = mesh_unfolder(mesh_flow)
            mesh_grid_X, mesh_grid_Y = torch.meshgrid(torch.arange(start=0, end=args.image_size[0], step=patch_size[0], dtype=torch.long),
                torch.arange(start=0, end=args.image_size[1], step=patch_size[1], dtype=torch.long),
                indexing="xy",
            )
            mesh_grid = torch.cat([mesh_grid_X.unsqueeze(0), mesh_grid_Y.unsqueeze(0)], dim=0)

            points_grid = torch.meshgrid(torch.arange(args.image_size[0]), torch.arange(args.image_size[1]), indexing='xy')
            points_grid = torch.cat([points_grid[0].unsqueeze(0), points_grid[1].unsqueeze(0)], dim=0)
            grid_unfolder = nn.Unfold(kernel_size=(args.image_size[0] // (args.mesh_size_3[0] - 1), args.image_size[1] // (args.mesh_size_3[1] - 1)), 
                stride=(args.image_size[0] // (args.mesh_size_3[0] - 1), args.image_size[1] // (args.mesh_size_3[1] - 1))
            )
            unfold_points_grid = grid_unfolder(points_grid)

            
            H1 = solve_DLT(delta_coors1, device=device, patch_size=patch_size)
            H2 = solve_DLT(delta_coors1 + delta_coors2, device=device, patch_size=patch_size)
            H3 = solve_DLT(delta_coors1 + delta_coors2 + delta_coors3, device=device, patch_size=patch_size)
            H1 = H_scale(H1, patch_size, args.batch_size, device)
            H2 = H_scale(H2, patch_size, args.batch_size, device)
            H3 = H_scale(H3, patch_size, args.batch_size, device)
            reference_imgs = input_tensor[:, 0:3, :, :]
            warp_input2_0 = spatial_transform(input_tensor[:, 3:6, :, :], H1, device=device)
            warp_input2_1 = spatial_transform(input_tensor[:, 3:6, :, :], H2, device=device)
            warp_input2_2 = spatial_transform(input_tensor[:, 3:6, :, :], H3, device=device)
            ones_mask = torch.ones_like(input_tensor[:, 3:6, :, :], dtype=torch.float32, device=device)
            warp_mask_0 = spatial_transform(ones_mask, H1, device=device)
            warp_mask_0[torch.where(warp_mask_0 < 0.4)] = 0
            warp_mask_1 = spatial_transform(ones_mask, H2, device=device)
            warp_mask_1[torch.where(warp_mask_1 < 0.4)] = 0
            warp_mask_2 = spatial_transform(ones_mask, H3, device=device)
            warp_mask_2[torch.where(warp_mask_2 < 0.4)] = 0
            loss1 = F.l1_loss(reference_imgs[warp_mask_0.bool()], warp_input2_0[warp_mask_0.bool()])
            loss2 = F.l1_loss(reference_imgs[warp_mask_1.bool()], warp_input2_1[warp_mask_1.bool()])
            loss3 = F.l1_loss(reference_imgs[warp_mask_2.bool()], warp_input2_2[warp_mask_2.bool()])
            loss = args.loss_weight[0] * loss1 + args.loss_weight[1] * loss2 + args.loss_weight[2] * loss3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss[0] += loss.item()
            running_loss[1] += loss1.item()
            running_loss[2] += loss2.item()
            running_loss[3] += loss3.item()
            print_loss += loss.item()
            
            if (i + 1) % print_every_iter == 0:
                logger.info('[{}, {}] running_loss = {:.5f}, learning_rate = {:.7f}'.format(epoch + 1, i + 1, print_loss / print_every_iter, optimizer.state_dict()['param_groups'][0]['lr']))                
                print_loss = 0.
            if (i + 1) % record_every_iter == 0:
                writer.add_scalar("loss/loss1", running_loss[1] / record_every_iter, tb_index)
                writer.add_scalar("loss/loss2", running_loss[2] / record_every_iter, tb_index)
                writer.add_scalar("loss/loss3", running_loss[3] / record_every_iter, tb_index)
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
        ssim_result = []
        psnr_result = []
        with torch.no_grad():
            for i, sample_batch in enumerate(val_loader):
                input_tensor = sample_batch[0].to(device)
                # input_tensor_aug = sample_batch[1].to(device)
                delta_coors1, delta_coors2, delta_coors3 = model(input_tensor)
                reference_imgs = input_tensor[:, 0:3, :, :]
                patch_size = 128.
                H3 = solve_DLT(delta_coors1 + delta_coors2 + delta_coors3, device=device, patch_size=patch_size)
                H3 = H_scale(H3, patch_size, args.batch_size, device)
                warp_target = spatial_transform(input_tensor[:, 3:6, :, :], H3, device=device)
                ones_mask = torch.ones_like(input_tensor[:, 3:6, :, :], dtype=torch.float32, device=device)
                warp_mask = spatial_transform(ones_mask, H3, device=device)
                warp_mask[torch.where(warp_mask < 0.4)] = 0
                warp_mask = warp_mask.bool()
                tensor1 = tensor2img2(reference_imgs, device=device) * warp_mask
                tensor2 = tensor2img2(warp_target, device=device) * warp_mask
                if i == 0:
                    writer.add_image("val/reference image",tensor1[0, ...], epoch)
                    writer.add_image("val/warped image",tensor2[0, ...], epoch)
                tensor1 = tensor1.float()
                tensor2 = tensor2.float()
                ssim_tensor = ssim(tensor1, tensor2, data_range=255, size_average=False)
                psnr_tensor = calc_psnr(tensor1, tensor2)
                ssim_result.append(ssim_tensor)
                psnr_result.append(psnr_tensor)
        psnr_result = torch.cat(psnr_result, dim=0)
        # avoid infinite psnr value result
        psnr_result = psnr_result[~torch.isinf(psnr_result)]
        psnr_value = torch.mean(psnr_result)
        ssim_value = torch.mean(torch.cat(ssim_result), dim=0)
        logger.info("Epoch {epoch} | psnr {psnr_value} | ssim {ssim_value}".format(epoch=epoch + 1, psnr_value=psnr_value, ssim_value=ssim_value))
        writer.add_scalar("metrics/psnr", psnr_value, epoch)
        writer.add_scalar("metrics/ssim", ssim_value, epoch)

    logger.info("training complete.")

if __name__ == "__main__":
    train(args=args)





