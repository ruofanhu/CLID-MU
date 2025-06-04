# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torchvision
import numpy as np
import math

from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from semilearn.datasets.utils import split_ssl_data
from PIL import Image
import json
import random
import torch

mean, std = {}, {}
mean['clothing1m'] = [0.6959, 0.6537, 0.6371]

std['clothing1m'] = [0.3113, 0.3192, 0.3214]


def get_clothing1m(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    train_labels ={}
    test_labels ={}
    
    # follow ELR 
    num_samples = int(64*3000)
    data_dir = os.path.join(data_dir, 'clothing1m')
    
    data_dir_imgs = os.path.join(data_dir,'images')
    # dsets_path =  os.path.join(data_dir,f'{args.seed}_{num_samples}_sets.pt')
    
    # if os.path.exists(dsets_path):
    #     lb_dset, eval_dset = torch.load(dsets_path)
    #     return lb_dset, eval_dset
        
    with open('%s/noisy_label_kv.txt'%data_dir,'r') as f:
        lines = f.read().splitlines()
        for l in lines:
            entry = l.split()           
            img_path = '%s/'%data_dir_imgs+entry[0][7:]
            
            train_labels[img_path] = int(entry[1])     
    
    with open('%s/clean_label_kv.txt'%data_dir,'r') as f:
        lines = f.read().splitlines()
        for l in lines:
            entry = l.split()           
            img_path = '%s/'%data_dir_imgs+entry[0][7:]
            test_labels[img_path] = int(entry[1])    
            
    train_imgs=[]
    with open('%s/noisy_train_key_list.txt'%data_dir,'r') as f:
        lines = f.read().splitlines()
        for l in lines:
            img_path = '%s/'%data_dir_imgs+l[7:]
            train_imgs.append(img_path)                                
    random.shuffle(train_imgs)
    
    class_num = torch.zeros(num_classes)
    train_imgs_ = []
    for impath in train_imgs:
        label = train_labels[impath] 
        if class_num[label]<(num_samples/14) and len(train_imgs_)<num_samples:
            train_imgs_.append(impath)
            class_num[label]+=1
    random.shuffle(train_imgs_) 
    
    targets = [train_labels[img_path] for img_path in train_imgs_]
    data = [Image.open(img_path).convert('RGB') for img_path in train_imgs_]

    # targets = [train_labels[img_path] for img_path in train_imgs]
    # data = [Image.open(img_path).convert('RGB') for img_path in train_imgs]
    # print(len(data))

    
    crop_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=224,padding=int(224*0.125),padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=224,padding=int(224*0.125),padding_mode='reflect'),
        RandAugment(2, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])
    # lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(args, data, targets, num_classes, 
    #                                                             lb_num_labels=num_labels,
    #                                                             ulb_num_labels=args.ulb_num_labels,
    #                                                             lb_imbalance_ratio=args.lb_imb_ratio,
    #                                                             ulb_imbalance_ratio=args.ulb_imb_ratio,
    #                                                             include_lb_to_ulb=include_lb_to_ulb)



    
    lb_count = [0 for _ in range(num_classes)]
    # ulb_count = [0 for _ in range(num_classes)]
    for c in targets:
        lb_count[c] += 1
    # for c in ulb_targets:
    #     ulb_count[c] += 1
    print("lb count: {}".format(lb_count))
    # print("ulb count: {}".format(ulb_count))
    # lb_count = lb_count / lb_count.sum()
    # ulb_count = ulb_count / ulb_count.sum()
    # args.lb_class_dist = lb_count
    # args.ulb_class_dist = ulb_count

    # if alg == 'fullysupervised' or alg == 'fullysupervised_lossnet':
    #     lb_data = data
    #     lb_targets = targets

    lb_dset = BasicDataset(alg, data, targets, num_classes, transform_weak, False, None, False)
    
    # ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_strong, False)
    print('already get lb_dset')
    test_imgs = []
    with open('%s/clean_test_key_list.txt'%data_dir,'r') as f:
        lines = f.read().splitlines()
        for l in lines:
            img_path = '%s/'%data_dir_imgs+l[7:]
            test_imgs.append(img_path)
    test_targets = [test_labels[img_path] for img_path in test_imgs]
    test_data = [Image.open(img_path).convert('RGB') for img_path in test_imgs]

    eval_dset = BasicDataset(alg, test_data, test_targets, num_classes, transform_val, False, None, False)
    
    # torch.save([lb_dset, eval_dset],dsets_path)
    
    return lb_dset, eval_dset
