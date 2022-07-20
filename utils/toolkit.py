import torch

def tensor2img(x:torch.Tensor) -> torch.Tensor:
    x *= 255
    x = torch.clamp(x, 0, 255)
    x = x.int()
    return x