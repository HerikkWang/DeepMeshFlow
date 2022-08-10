import torch
import os
from torchvision import transforms
from PIL import Image
from typing import Tuple
import argparse

class align_dataset(torch.utils.data.Dataset):
    def __init__(self, args:argparse.ArgumentParser, validation:bool=False) -> None:
        super().__init__()
        # self.dir = args.data_folder
        self.stage = args.stage
        if self.stage == 'train':
            self.dir = args.train_data
        else:
            self.dir = args.test_data
        if validation:
            self.dir = args.test_data
        self.data_inverse = args.data_inverse
        self.data_random_inverse = args.data_random_inverse
        self.brightness = args.brightness
        self.contrast = args.contrast
        self.saturation = args.saturation

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_list_1 = sorted(os.listdir(os.path.join(self.dir, "input1")))
        input_list_2 = sorted(os.listdir(os.path.join(self.dir, "input2")))
        img1_path = os.path.join(self.dir, "input1", input_list_1[idx])
        img2_path = os.path.join(self.dir, "input2", input_list_2[idx])
        if self.data_inverse:
            input_list_1 = sorted(os.listdir(os.path.join(self.dir, "input2")))
            input_list_2 = sorted(os.listdir(os.path.join(self.dir, "input1")))
            img1_path = os.path.join(self.dir, "input2", input_list_1[idx])
            img2_path = os.path.join(self.dir, "input1", input_list_2[idx])
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        img_name = input_list_1[idx]

        img2pair = self.data_augmentation(img1, img2, random_data_inverse=self.data_random_inverse)
        return img2pair, img_name
    
    def __len__(self):
        return len(os.listdir(os.path.join(self.dir, "input1")))
    
    def data_augmentation(self, img1: Image.Image, img2: Image.Image, random_data_inverse=False) -> Tuple[torch.Tensor]:
        p_vertical = torch.randint(low=0, high=2, size=(1,))[0]
        p_horizontal = torch.randint(low=0, high=2, size=(1,))[0]
        transform_flip = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p_horizontal),
            transforms.RandomVerticalFlip(p=p_vertical),
        ])
        img1 = transform_flip(img1)
        img2 = transform_flip(img2)
        transform_jitter = transforms.Compose([
            transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation),
            transforms.ToTensor(),
        ])     
        img1_aug = transform_jitter(img1)
        img2_aug = transform_jitter(img2)
        img1 = transforms.ToTensor()(img1)
        img2 = transforms.ToTensor()(img2)
        # img1 = torch.mean(img1, dim=1, keepdim=True)
        # img2 = torch.mean(img2, dim=1, keepdim=True)
        # img1_aug = torch.mean(img1_aug, dim=1, keepdim=True)
        # img2_aug = torch.mean(img2_aug, dim=1, keepdim=True)
        if random_data_inverse:
            p_inverse = torch.rand(size=(1,))[0]
            if p_inverse >= 0.5:
                origin_pair = torch.cat([img1, img2], dim=0)
                aug_pair = torch.cat([img1_aug, img2_aug], dim=0)
            else:
                origin_pair = torch.cat([img2, img1], dim=0)
                aug_pair = torch.cat([img2_aug, img1_aug], dim=0)
        else:
            origin_pair = torch.cat([img1, img2], dim=0)
            aug_pair = torch.cat([img1_aug, img2_aug], dim=0)
       
        return torch.cat([origin_pair, aug_pair], dim=0)
        