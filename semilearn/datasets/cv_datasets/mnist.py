# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torchvision
import numpy as np
import math

from torchvision import transforms, datasets
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from semilearn.datasets.utils import split_ssl_data

class Customizer_Dataset():
    def __init__(self, dataset, transforms, labels = None, index=None):

        self.dataset = dataset
        self.labels = labels
        self.transforms = transforms

        if index is None:
            self.index = np.arange(len(self.dataset))
        else:
            self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        _idx = self.index[idx]

        # data_1 = self.transforms(self.dataset[_idx][0])
        data_1 = self.transforms(self.dataset[_idx][0])


        if self.labels is not None:
            label = self.labels[_idx]
        else:
            label = self.dataset[_idx][1]

        return {'idx_lb': _idx, 'x_lb': data_1, 'y_lb': label,'y_true':label} 


def get_mnist(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    
    data_dir = os.path.join(data_dir, name.lower())
    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, train=True, download=True)
    data, targets = dset.data, dset.targets
#     dset = datasets.MNIST(data_dir, train=True, download=True) #60000
    data, targets = dset.data, dset.targets
    crop_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomHorizontalFlip(),
        RandAugment(5, 5, exclude_color_aug=True,fixmatch=False),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(args, data, targets, num_classes, 
                                                                lb_num_labels=num_labels,
                                                                ulb_num_labels=args.ulb_num_labels,
                                                                lb_imbalance_ratio=args.lb_imb_ratio,
                                                                ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                include_lb_to_ulb=include_lb_to_ulb)
    
    lb_count = [0 for _ in range(num_classes)]
    ulb_count = [0 for _ in range(num_classes)]
    for c in lb_targets:
        lb_count[c] += 1
    for c in ulb_targets:
        ulb_count[c] += 1
    print("lb count: {}".format(lb_count))
    print("ulb count: {}".format(ulb_count))
    # lb_count = lb_count / lb_count.sum()
    # ulb_count = ulb_count / ulb_count.sum()
    # args.lb_class_dist = lb_count
    # args.ulb_class_dist = ulb_count

    if alg == 'fullysupervised':
        lb_data = data
        lb_targets = targets
        # if len(ulb_data) == len(data):
        #     lb_data = ulb_data 
        #     lb_targets = ulb_targets
        # else:
        #     lb_data = np.concatenate([lb_data, ulb_data], axis=0)
        #     lb_targets = np.concatenate([lb_targets, ulb_targets], axis=0)
    
    # output the distribution of labeled data for remixmatch
    # count = [0 for _ in range(num_classes)]
    # for c in lb_targets:
    #     count[c] += 1
    # dist = np.array(count, dtype=float)
    # dist = dist / dist.sum()
    # dist = dist.tolist()
    # out = {"distribution": dist}
    # output_file = r"./data_statistics/"
    # output_path = output_file + str(name) + '_' + str(num_labels) + '.json'
    # if not os.path.exists(output_file):
    #     os.makedirs(output_file, exist_ok=True)
    # with open(output_path, 'w') as w:
    #     json.dump(out, w)
    
    
#     test_data, test_targets = dset.data, dset.targets
    
#     eval_dset = BasicDataset(alg, test_data, test_targets, num_classes, transform_val, False, None, False)
    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, train=False, download=True)
    eval_dset = Customizer_Dataset(dset, transform_val)
    lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, None, False)


    ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_strong, False)

#     dset = getattr(torchvision.datasets, name.upper())
#     dset = dset(data_dir, train=False, download=True)

    return  lb_dset, ulb_dset, eval_dset
