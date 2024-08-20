# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import numpy as np

from semilearn.datasets.utils import split_ssl_data
from .datasetbase import BasicDataset
from collections import Counter


def get_json_dset(args, alg='fixmatch', dataset='acmIb', num_labels=40, num_classes=20, data_dir='./data', index=None, include_lb_to_ulb=True, onehot=False):
        """
        get_ssl_dset split training samples into labeled and unlabeled samples.
        The labeled data is balanced samples over classes.
        
        Args:
            num_labels: number of labeled data.
            index: If index of np.array is given, labeled data is not randomly sampled, but use index for sampling.
            include_lb_to_ulb: If True, consistency regularization is also computed for the labeled data.
            strong_transform: list of strong transform (RandAugment in FixMatch)
            onehot: If True, the target is converted into onehot vector.
            
        Returns:
            BasicDataset (for labeled data), BasicDataset (for unlabeld data)
        """
        json_dir = os.path.join(data_dir, dataset)
        
        # Supervised top line using all data as labeled data.
        with open(os.path.join(json_dir,'train.json'),'r') as json_data:
            train_data = json.load(json_data)
            train_sen_list = []
            train_label_list = []
            for idx in train_data:
                train_sen_list.append((train_data[idx]['ori'],train_data[idx]['aug_0'],train_data[idx]['aug_1']))
                train_label_list.append(int(train_data[idx]['label']))
        with open(os.path.join(json_dir,'dev.json'),'r') as json_data:
            dev_data = json.load(json_data)
            dev_sen_list = []
            dev_label_list = []
            for idx in dev_data:
                dev_sen_list.append((dev_data[idx]['ori'],'None','None'))
                dev_label_list.append(int(dev_data[idx]['label']))
        with open(os.path.join(json_dir,'test.json'),'r') as json_data:
            test_data = json.load(json_data)
            test_sen_list = []
            test_label_list = []
            for idx in test_data:
                test_sen_list.append((test_data[idx]['ori'],'None','None'))
                test_label_list.append(int(test_data[idx]['label']))
        dev_dset = BasicDataset(alg, dev_sen_list, dev_label_list, num_classes, False, onehot)
        test_dset = BasicDataset(alg, test_sen_list, test_label_list, num_classes, False, onehot)
        if alg == 'fullysupervised':
            lb_dset = BasicDataset(alg, train_sen_list, train_label_list, num_classes, False,onehot)
            return lb_dset, None, dev_dset, test_dset
        
        lb_sen_list, lb_label_list, ulb_sen_list, ulb_label_list = split_ssl_data(args, train_sen_list, train_label_list, num_classes, 
                                                                    lb_num_labels=num_labels,
                                                                    ulb_num_labels=args.ulb_num_labels,
                                                                    lb_imbalance_ratio=args.lb_imb_ratio,
                                                                    ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                    include_lb_to_ulb=include_lb_to_ulb)

        # output the distribution of labeled data for remixmatch
        count = [0 for _ in range(num_classes)]
        for c in train_label_list:
            count[c] += 1
        dist = np.array(count, dtype=float)
        dist = dist / dist.sum()
        dist = dist.tolist()
        out = {"distribution": dist}
        output_file = r"./data_statistics/"
        output_path = output_file + str(dataset) + '_' + str(num_labels) + '.json'
        if not os.path.exists(output_file):
            os.makedirs(output_file, exist_ok=True)
        with open(output_path, 'w') as w:
            json.dump(out, w)
        
        lb_dset = BasicDataset(alg, lb_sen_list, lb_label_list, num_classes, False, onehot)
        ulb_dset = BasicDataset(alg, ulb_sen_list, ulb_label_list, num_classes, True, onehot)
        return lb_dset, ulb_dset, dev_dset, test_dset


def generate_mv(labels):
    labels=[int(i) for i in labels]
    num_ab=labels.count(-1)
    if num_ab==len(labels):
        return -1
    count = Counter(labels)
    
    # Remove -1 from the count if it exists
    if -1 in count:
        del count[-1]
    
    # Find the most common element and its frequency
    if count:
        most_common = count.most_common(1)[0][0]
        return most_common
    
def get_json_data_weak(args, alg='fullysupervised_lossnet', dataset='agnews', num_classes=10, data_dir='./data', index=None, include_lb_to_ulb=True, onehot=False):
        json_dir = os.path.join(data_dir, dataset)
        
        with open(os.path.join(json_dir,'train.json'),'r') as json_data:
            train_data = json.load(json_data)
            
            train_lb_id_list = []
            train_lb_sen_list = []
            train_lb_label_list = []
            train_lb_w_label_list= []
            train_ulb_id_list = []
            train_ulb_sen_list = []
            train_ulb_label_list = []
            for idx in train_data:
                mv_label = generate_mv(train_data[idx]['weak_labels'])
                if mv_label==-1:
                    train_ulb_id_list.append(int(idx))
                    train_ulb_sen_list.append((train_data[idx]['data']['text'],'None','None'))
                    train_ulb_label_list.append(int(train_data[idx]['label']))
                else:
                    train_lb_id_list.append(int(idx))
                    train_lb_sen_list.append((train_data[idx]['data']['text'],'None','None'))
                    train_lb_label_list.append(int(train_data[idx]['label']))                    
                    train_lb_w_label_list.append(mv_label)

        with open(os.path.join(json_dir,'valid.json'),'r') as json_data:
            dev_data = json.load(json_data)
            
            dev_lb_id_list = []
            dev_lb_sen_list = []
            dev_lb_label_list = []
            dev_lb_w_label_list= []
            dev_ulb_id_list = []
            dev_ulb_sen_list = []
            dev_ulb_label_list = []
            for idx in dev_data:
                # train_label_list.append(int(train_data[idx]['label'])) 
                mv_label = generate_mv(dev_data[idx]['weak_labels'])
                if mv_label==-1:
                    dev_ulb_id_list.append(int(idx))
                    dev_ulb_sen_list.append((dev_data[idx]['data']['text'],'None','None'))
                    dev_ulb_label_list.append(int(dev_data[idx]['label']))
                else:
                    dev_lb_id_list.append(int(idx))
                    dev_lb_sen_list.append((dev_data[idx]['data']['text'],'None','None'))
                    dev_lb_label_list.append(int(dev_data[idx]['label']))                    
                    dev_lb_w_label_list.append(mv_label) 
                
        with open(os.path.join(json_dir,'test.json'),'r') as json_data:
            test_data = json.load(json_data)
            test_id_list = []
            test_sen_list = []
            test_label_list = []
            for idx in test_data:
                test_id_list.append(int(idx))
                test_sen_list.append((test_data[idx]['data']['text'],'None','None'))
                test_label_list.append(int(test_data[idx]['label']))   
        dev_l_dset = BasicDataset(alg, dev_lb_sen_list,dev_lb_w_label_list,dev_lb_label_list, num_classes, False, onehot)
        dev_u_dset = BasicDataset(alg, dev_ulb_sen_list,dev_ulb_label_list,dev_ulb_label_list, num_classes, False, onehot)
        train_l_dset = BasicDataset(alg, train_lb_sen_list,train_lb_w_label_list,train_lb_label_list, num_classes, False, onehot)
        train_u_dset = BasicDataset(alg, train_ulb_sen_list,train_ulb_label_list,train_ulb_label_list, num_classes, False, onehot)

        test_dset = BasicDataset(alg, test_sen_list,test_label_list,test_label_list, num_classes, False, onehot)
        return train_l_dset, train_u_dset, dev_l_dset, dev_u_dset, test_dset