# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torchvision
import numpy as np
import math

from torchvision import transforms
from .datasetbase_toy import BasicDataset_toy
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from semilearn.datasets.utils import split_ssl_data
import matplotlib.pyplot as plt
import torch


def two_ball(num_labels, num_classes,args):
    np.random.seed(args.seed)
    n_samples = 5000  # number of samples to generate

#     mean_1 = [1, 0, 0]
#     cov_1 = [[0.25, 0,0], [0, 0.25,0],[0, 0,0.25]]  # diagonal covariance

#     mean_2 = [6, 0, 0]
#     cov_2 = [[0.25, 0,0], [0, 0.25,0],[0, 0,0.25]] # diagonal covariance

    mean_1 = [1, 0]
    cov_1 = [[0.25, 0], [0, 0.25]]  # diagonal covariance

    mean_2 = [6, 0]
    cov_2 = [[0.25, 0], [0, 0.25]] # diagonal covariance
    
    x_1 = np.random.multivariate_normal(mean_1, cov_1, int(n_samples/2)).T

    x_2 = np.random.multivariate_normal(mean_2, cov_2, int(n_samples/2)).T

    y_1 =np.ones(int(n_samples/2))
    y_2 =np.zeros(int(n_samples/2))

    x=np.concatenate((x_1, x_2), axis=1).T
    y=np.concatenate((y_1, y_2), axis=0)
    print(x.shape,y)
    plt.scatter(*x.T,c=y,cmap=plt.cm.Accent)
    # plt.axvline(60, color='red',linestyle='dashed')

    plt.title("Generated  two clusters data (all)");
    plt.axis('equal')
    plt.ylim(-3,2.5)
    # plt.xlim(-4,4)
    plt.show()


      # fraction of data to drop labels

    _index=range(n_samples)
    unlabeled = np.random.choice(range(n_samples), size=n_samples-num_labels, replace=False)

    lab_idx= np.array(list((set(_index)-set(unlabeled))))
     
    return x, y, lab_idx, unlabeled

def half_moon(num_labels, num_classes,args):
    np.random.seed(args.seed)
    n_samples = 5000  # number of samples to generate
    from sklearn import datasets
#     mean_1 = [1, 0, 0]
#     cov_1 = [[0.25, 0,0], [0, 0.25,0],[0, 0,0.25]]  # diagonal covariance

#     mean_2 = [6, 0, 0]
#     cov_2 = [[0.25, 0,0], [0, 0.25,0],[0, 0,0.25]] # diagonal covariance

    
    
    
    x,y = datasets.make_moons(n_samples=n_samples, shuffle=True, noise=0, random_state=args.seed)
#     y = np.reshape(y, (len(y),1))


      # fraction of data to drop labels

    _index=range(n_samples)
    unlabeled = np.random.choice(range(n_samples), size=n_samples-num_labels, replace=False)

    lab_idx= np.array(list((set(_index)-set(unlabeled))))
     
    return x, y, lab_idx, unlabeled





def get_toy(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    
    
#     x, y, lab_idx, unlabeled = two_ball(num_labels, num_classes,args)
    
    x, y, lab_idx, unlabeled = half_moon(num_labels, num_classes,args)

    
    
    x_l_all =x[lab_idx]
    x_u_all = x[unlabeled]

    y_l_all = y[lab_idx] 
    y_u_all = y[unlabeled]
    assert len(x_l_all)==num_labels

    

    lb_data = torch.from_numpy(x_l_all).type(torch.float)
    ulb_data = torch.from_numpy(x).type(torch.float)
    lb_targets = torch.from_numpy(y_l_all).type(torch.int64) #noisy labels
#     y_l_all_ = torch.from_numpy(y_l_all_).type(torch.float) #clean labels

    ulb_targets = torch.from_numpy(y).type(torch.int64)
    
    all_data=torch.from_numpy(x).type(torch.float)
    all_targets=torch.from_numpy(y).type(torch.int64)

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
    crop_size=32 
    crop_ratio=0.7
    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
#         transforms.Normalize(mean[name], std[name])
    ])    
    
    lb_dset = BasicDataset_toy(alg, lb_data, lb_targets, num_classes, transform_weak, False, None, False)
    ulb_dset = BasicDataset_toy(alg, ulb_data, ulb_targets, num_classes, transform_weak,True, None, False)

#     ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak,True, None, False)

#     dset = getattr(torchvision.datasets, name.upper())
#     dset = dset(data_dir, train=False, download=True)
#     test_data, test_targets = dset.data, dset.targets
    eval_dset = BasicDataset_toy(alg, all_data, all_targets, num_classes, None, False, None, False)
    
    return lb_dset, ulb_dset, eval_dset
