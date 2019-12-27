import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, augment=False):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.max_objects = 100
        self.batch_count = 0
        self.augment = augment
        
    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))
        
        
        h, w = img.shape[1:]
        h_factor, w_factor = (h, w)
        # Pad to square resolution
        # img, pad = pad_to_square(img, 0)
        # _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()
        targets = {}
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5).astype(np.float32))
            if self.augment:
                if np.random.random() < 0.5:
                    img, boxes[:, 1:] = horisontal_flip(img, boxes[:, 1:])
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            boxes[:, 1] = x1
            boxes[:, 2] = y1
            boxes[:, 3] = x2
            boxes[:, 4] = y2
            
            targets['boxes'] = boxes[:, 1:]
            targets['labels'] = boxes[:, 0].long() + 1
            if len(targets['boxes'].shape) == 1:
                targets['boxes'] = targets['boxes'].unsqueeze(0)
            # targets['image_ids'] = boxes[:, 0].new(*boxes[:, 0].shape).zeros_()

        # # Apply augmentations
        # if self.augment:
        #     if np.random.random() < 0.5:
        #         img, targets = horisontal_flip(img, targets)

        return img, targets

    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [target for target in targets if target is not None]
        # Add sample index to targets
        for i, target in enumerate(targets):
            target['image_ids'] = i
        # targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        # if self.multiscale and self.batch_count % 10 == 0:
        #     self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = [img for img in imgs if img is not None]
        self.batch_count += 1
        return imgs, targets

    def __len__(self):
        return len(self.img_files)
