from torch.utils.tensorboard import SummaryWriter
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
img = Image.open("im_test/aug0.jpg").convert("RGB")
im_tensor = transforms.ToTensor()(img)
# img[0] = np.arange(0, 10000).reshape(100, 100) / 10000
# img[1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

# img_HWC = np.zeros((100, 100, 3))
# img_HWC[:, :, 0] = np.arange(0, 10000).reshape(100, 100) / 10000
# img_HWC[:, :, 1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

writer = SummaryWriter()
writer.add_image('my_image', im_tensor, 0)

# If you have non-default dimension setting, set the dataformats argument.
# writer.add_image('my_image_HWC', img_HWC, 0, dataformats='HWC')
writer.close()