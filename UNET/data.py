import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path, mode=1):

        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)
        self.mode = mode

    def __getitem__(self, index):
        """reading image"""
        image = cv2.imread(self.images_path[index], self.mode)
        image = image / 255.0  # (512, 512, 3)
        if self.mode:
            image = np.transpose(image, (2, 0, 1))# (3, 512, 512)
        else:
            image = np.expand_dims(image, axis=0) # (1, 512, 512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        mask = cv2.imread(self.masks_path[index], 0)
        mask = mask / 255.0  # (512, 512)
        mask = np.expand_dims(mask, axis=0)  # (1, 512, 512)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        return image, mask

    def __len__(self):
        return self.n_samples
