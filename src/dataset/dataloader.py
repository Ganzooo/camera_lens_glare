#!/usr/bin/env python3
# coding: utf-8

from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torch
import cv2
from augmentations import Compose, RandomHorizontallyFlip, RandomRotate, Scale, RandomVerticallyFlip


def load_dataset(data_path, batch_size, distributed, train_valid_split_weight=0.9, resize_size=(512,512)):

    _dataloader = DataLoaderImg(data_path, mode='train', resize_size=resize_size)

    train_size = int(train_valid_split_weight * len(_dataloader))
    val_size = len(_dataloader) - train_size

    print("train-size",train_size)
    print("val-size",val_size)
    train_dataset, val_dataset = random_split(_dataloader, [train_size, val_size])
    
    test_dataset = DataLoaderImg(data_path, mode='test')
    
    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

class DataLoaderImg(Dataset):
    def __init__(self, data_path, mode='train', transform=True, resize_size=(512,512)):
        self.data_path = data_path
        self.transform = transform
        self.target_size = resize_size
        self.dataset_type = mode
        augmentations = Compose([RandomRotate(10), RandomHorizontallyFlip(0.5), RandomVerticallyFlip(0.5)])
        #augmentations = {"hflip":0.5, "vflip":0.2}
        self.augmentations = augmentations
        if self.dataset_type != 'test':
            self.in_feature_paths = list(sorted(Path(self.data_path).glob("train/train_input_img/*.png")))
            self.target_feature_paths = list(sorted(Path(self.data_path).glob("train/train_label_img/*.png")))
        else:
            self.test_feature_paths = list(sorted(Path(self.data_path).glob("test_input_img/*.png")))

    def __len__(self):
        if self.dataset_type == 'test':
            return len(self.test_feature_paths)
        return len(self.in_feature_paths)
    
    def __transform__(self, img):
        img = cv2.resize(img, (self.target_size[0], self.target_size[1]))
        img = img / 255.0
        img = img.transpose(2, 0, 1)

        return torch.from_numpy(img).float()


    def __getitem__(self, idx):
        if self.dataset_type == 'test':
            img = cv2.cvtColor(cv2.imread(str(self.test_feature_paths[idx])), cv2.COLOR_BGR2RGB)

            return self.__transform__(img), self.test_feature_paths[idx].name

        img = cv2.cvtColor(cv2.imread(str(self.in_feature_paths[idx])), cv2.COLOR_BGR2RGB)
        lbl = cv2.cvtColor(cv2.imread(str(self.target_feature_paths[idx])), cv2.COLOR_BGR2RGB)
        
        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)
        return self.__transform__(img), self.__transform__(lbl), self.in_feature_paths[idx].name
