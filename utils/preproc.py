# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

# Modifications made by Shun Miura(https://github.com/Miura-code)

import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from utils import setting

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img
    
class GrayToRGB:
    # input is PIL image
    def __init__(self) -> None:
        pass

    def __call__(self, x):
        x = x.convert("L").convert("RGB")
        return x

def data_transforms(dataset, cutout_length):
    dataset = dataset.lower()
    if dataset == 'cifar10':
        MEAN = setting.CIFAR10_MEAN
        STD = setting.CIFAR10_STD
        transf = [
            transforms.RandomCrop(32, padding=4, fill=128),
            transforms.RandomHorizontalFlip()
        ]
    elif dataset == 'cifar100':
        MEAN = setting.CIFAR100_MEAN
        STD = setting.CIFAR100_STD
        transf = [
            transforms.RandomCrop(32, padding=4, fill=128),
            transforms.RandomHorizontalFlip()
        ]
    elif dataset == 'mnist':
        MEAN = setting.MNIST_MEAN
        STD = setting.MNIST_STD
        transf = [
            transforms.Resize(size=(32, 32)),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
            GrayToRGB()
        ]
    elif dataset == 'fashionmnist':
        MEAN = setting.FASHIONMNIST_MEAN
        STD = setting.FASHIONMNIST_STD
        transf = [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
            transforms.RandomVerticalFlip()
        ]
    else:
        raise ValueError('not expected dataset = {}'.format(dataset))

    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]

    train_transform = transforms.Compose(transf + normalize)
    valid_transform = transforms.Compose(normalize)

    if cutout_length > 0:
        train_transform.transforms.append(Cutout(cutout_length))

    return train_transform, valid_transform


def data_transforms_advanced(dataset, cutout_length):
    dataset = dataset.lower()
    if dataset == 'cifar10':
        MEAN = setting.CIFAR10_MEAN
        STD = setting.CIFAR10_STD
        transf = [
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip()
        ]
    elif dataset == 'cifar100':
        MEAN = setting.CIFAR100_MEAN
        STD = setting.CIFAR100_STD
        transf = [
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip()
        ]
    elif dataset == 'mnist':
        MEAN = setting.MNIST_MEAN
        STD = setting.MNIST_STD
        transf = [
            transforms.Resize(size=(32, 32)),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
            GrayToRGB()
        ]
    elif dataset == 'fashionmnist':
        MEAN = setting.FASHIONMNIST_MEAN
        STD = setting.FASHIONMNIST_STD
        transf = [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
            transforms.RandomVerticalFlip()
        ]
    else:
        raise ValueError('not expected dataset = {}'.format(dataset))

    normalize = [
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.Normalize(MEAN, STD)
    ]

    train_transform = transforms.Compose(transf + normalize)
    valid_transform = transforms.Compose(normalize)

    if cutout_length > 0:
        train_transform.transforms.append(Cutout(cutout_length))

    return train_transform, valid_transform