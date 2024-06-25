# -*- coding: utf-8 -*-
# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.
import os
import torchvision.datasets as dset
import torchvision.transforms as transforms

import utils.preproc as prep
from utils.preproc import Cutout


def get_data(dataset, data_path, cutout_length, validation):
    dataset = dataset.lower()
    if dataset == 'cifar10':
        dset_cls = dset.CIFAR10
        n_classes = 10
    elif dataset == 'cifar100':
        dset_cls = dset.CIFAR100
        n_classes = 100
    elif dataset == 'mnist':
        dset_cls = dset.MNIST
        n_classes = 10
    elif dataset == 'imagenet':
        return get_imagenet(dataset, data_path, cutout_length, validation)
    else:
        raise ValueError(dataset)

    trn_transform, val_transform = prep.data_transforms(dataset, cutout_length)
    trn_data = dset_cls(root=data_path, train=True, download=True, transform=trn_transform)

    # assuming shape is NHW or NHWC
    shape = trn_data.data.shape
    # input_channels = 3 if len(shape) == 4 else 1
    input_channels = 3
    assert shape[1] == shape[2], "not expected shape = {}".format(shape)
    input_size = shape[1]

    ret = [input_size, input_channels, n_classes, trn_data]
    if validation:  # append validation data
        ret.append(dset_cls(root=data_path, train=False, download=True, transform=val_transform))

    return ret


def get_imagenet(dataset, data_path, cutout_length, validation):
    dataset = dataset.lower()
    traindir = os.path.join(data_path,'imagenet/train/')
    validdir = os.path.join(data_path,'imagenet/val/')

    if not os.path.isdir(os.path.dirname(traindir)):
        os.makedirs(os.path.dirname(traindir), exist_ok=True)
    if not os.path.isdir(os.path.dirname(validdir)):
        os.makedirs(os.path.dirname(validdir), exist_ok=True)

    CLASSES = 1000
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if cutout_length == 0:
        train_data = dset.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
    elif cutout_length > 0:
        train_data = dset.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                Cutout(cutout_length),
            ]))
    valid_data = dset.ImageFolder(
        validdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    trn_data = train_data
    val_data = valid_data
    input_channels = 3
    input_size = 224
    ret = [input_size, input_channels, CLASSES, trn_data]
    if validation:
        ret.append(val_data)

    return ret