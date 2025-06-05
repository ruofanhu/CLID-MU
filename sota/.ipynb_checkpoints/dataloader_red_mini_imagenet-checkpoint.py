from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
import torchvision.transforms.functional as F
#from torchnet.meter import AUCMeter
import codecs
from collections import defaultdict

def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class red_mini_imagenet_dataset(Dataset): 
    def __init__(self, data, targets, transform, mode, pred=[], probability=[]): 
        
        self.transform = transform
        self.mode = mode

     
        if self.mode == 'all' or self.mode == 'test':
            self.data = data
            self.targets = targets
        elif self.mode == 'meta_clid':
            self.data = data
            self.targets = targets
            # class_to_indices = defaultdict(list)
            # for idx, target in enumerate(targets):
            #     class_to_indices[target].append(idx)
            
            # # Initialize lists to store the sampled data and targets
            # sampled_data = []
            # sampled_targets = []
            
            # # Iterate over each class and sample 100 indices
            # for cls, indices in class_to_indices.items():
            #     # Randomly sample 100 indices (or all indices if fewer than 100)
            #     sampled_indices = random.sample(indices, min(100, len(indices)))
            
            #     # Add the sampled data and targets
            #     sampled_data.extend(data[i] for i in sampled_indices)
            #     sampled_targets.extend(targets[i] for i in sampled_indices)

            # self.data = sampled_data
            # self.targets = sampled_targets        
        else:                   
            if self.mode == "labeled" or self.mode == "meta":
                pred_idx = pred.nonzero()[0]
                self.probability = [probability[i] for i in pred_idx]   
                    
            elif self.mode == "unlabeled":
                pred_idx = (1-pred).nonzero()[0]                                               
                self.probability = [probability[i] for i in pred_idx]
            print('len(data)',len(data))
            self.data = [data[i] for i in pred_idx]
            self.targets = [targets[i] for i in pred_idx]                          
            print("%s data has a size of %d"%(self.mode,len(self.targets)))            
        assert len(self.data) == len(self.targets)
    def __getitem__(self, index):
        # if self.mode=='meta':
        #     img, target, prob = self.data[index], self.targets[index], self.probability[index]
        #     img = self.transform(img) 
        #     return img, target, prob            

        if self.mode=='labeled':
            img, target, prob = self.data[index], self.targets[index], self.probability[index]
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2, target, prob            
        elif self.mode=='unlabeled':
            img, target, prob = self.data[index], self.targets[index], self.probability[index]
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2, target, prob
        elif self.mode=='all':
            img, target = self.data[index], self.targets[index]
            img = self.transform(img)           
            return img, target, index        
        elif self.mode=='meta_clid':
            img, target = self.data[index], self.targets[index]
            img = self.transform(img)           
            return img, target        
        elif self.mode=='test':
            img, target = self.data[index], self.targets[index]
            img = self.transform(img)            
            return img, target
           
    def __len__(self):
        return len(self.data)


class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return F.rotate(x, angle)

        
        
class red_mini_imagenet_dataloader():  
    def __init__(self, split_file, batch_size, num_workers, root_dir):

        self.split_file = split_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        # self.transform_train = transforms.Compose([
        #         # transforms.Resize((32, 32), interpolation=2),
        #         transforms.RandomCrop(32, padding=4),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Lambda(lambda x: 2 * (x - 0.5))  # Scale to [-1, 1]
        #         # transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        #     ]) 
        self.transform_train = transforms.Compose([
                #transforms.Resize((32, 32), interpolation=2),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ]) 
        self.transform_test = transforms.Compose([
                #transforms.Resize((32, 32), interpolation=2),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])   
        # self.transform_test = transforms.Compose([
        #     transforms.CenterCrop(32),  # Resize with crop or pad to the target size
        #     transforms.ToTensor(),              # Convert to tensor and normalize to [0, 1]
        #     transforms.Lambda(lambda x: (x * 255.0 - 127.5) / 127.5)  # Scale to [-1, 1]
        # ])        
        
        # load data first
        
        # data = []
        # targets = [] 
        # print('load all data into memory ....')
        # with codecs.open(split_file, 'r', 'utf-8') as rf:
        #     for line in rf:
        #         temp = line.strip().split(' ')
        #         image_name = temp[0]
        #         target = temp[1]
        #         img_path = '{}/{}/{}'.format(self.root_dir, target, image_name)
                
        #         img = Image.open(img_path).convert('RGB')
        #         data.append(img)
        #         targets.append(int(target))
        # print('done ....')
        if self.split_file=='noise_0.4':
            data,targets=torch.load('/home/rhu/r_work/VRI/VRI_DivideMix/data/red_mini_imagenet_s32/noise_0.4_first.pt')
        if self.split_file=='noise_0.8':
            data,targets=torch.load('/home/rhu/r_work/VRI/VRI_DivideMix/data/red_mini_imagenet_s32/noise_0.8.pt')
        else:
            data,targets=torch.load('/home/rhu/r_work/VRI/VRI_DivideMix/data/red_mini_imagenet_s32/clean_val_first.pt')
        
        self.data=[Image.fromarray((d*255).astype(np.uint8)) for d in data]

        self.targets = torch.tensor(targets, dtype=torch.long).squeeze()
        
    def run(self,mode,pred=[],prob=[]):
        if mode=='warmup':
            dataset = red_mini_imagenet_dataset(self.data, self.targets, transform=self.transform_train, mode="all")                
            trainloader = DataLoader(
                dataset=dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
        elif mode=='meta_clid':
            dataset = red_mini_imagenet_dataset(self.data, self.targets, transform=self.transform_train, mode="meta_clid")                
            trainloader = DataLoader(
                dataset=dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
        elif mode=='meta':
            meta_dataset = red_mini_imagenet_dataset(self.data, self.targets, transform=self.transform_train, mode="meta", pred=pred, probability=prob)   
            meta_trainloader = DataLoader(
                dataset=meta_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)   
            return meta_trainloader
        elif mode=='labeled':
            labeled_dataset = red_mini_imagenet_dataset(self.data, self.targets, transform=self.transform_train, mode="labeled", pred=pred, probability=prob)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)   
            return labeled_trainloader
        elif mode=='train':
            labeled_dataset = red_mini_imagenet_dataset(self.data, self.targets, transform=self.transform_train, mode="labeled", pred=pred, probability=prob)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = red_mini_imagenet_dataset(self.data, self.targets, transform=self.transform_train, mode="unlabeled", pred=pred, probability=prob)                    
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)     
            return labeled_trainloader, unlabeled_trainloader

        elif mode=='test':
            test_dataset = red_mini_imagenet_dataset(self.data, self.targets, transform=self.transform_test, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = red_mini_imagenet_dataset(self.data, self.targets, transform=self.transform_test, mode='all')      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader