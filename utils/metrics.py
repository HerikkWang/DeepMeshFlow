import torch

def tensor2img(x:torch.Tensor, device: torch.device, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) -> torch.Tensor:
    mean = torch.tensor(mean, dtype=torch.float32, device=device)
    std = torch.tensor(std, dtype=torch.float32, device=device)
    mean = mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    std = std.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    x = (x * std + mean) * 255
    x = torch.clamp(x, 0, 255)
    x = x.int()
    return x

def tensor2img2(x:torch.Tensor, device: torch.device) -> torch.Tensor:
    x = (x + 1) * 127.5
    x = torch.clamp(x, 0, 255)
    x = x.int()
    return x

def calc_psnr(tensor1:torch.tensor, tensor2:torch.tensor) -> torch.Tensor:
    mse = torch.mean(((tensor1 - tensor2) ** 2).type(torch.float32), dim=(1, 2, 3))
    psnr = 10 * torch.log10(255 * 255 / mse)
    return psnr

def calc_mae(tensor1:torch.tensor, tensor2:torch.tensor) -> torch.Tensor:
    mae = torch.mean(torch.abs(tensor1 - tensor2))
    return mae

# def calc_ssim(tensor1:torch.tensor, tensor2:torch.tensor) -> torch.Tensor:



