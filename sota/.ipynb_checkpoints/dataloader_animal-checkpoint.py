#https://github.com/arpit2412/InstanceGM/blob/main/dataloader_animal10N.py
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import random
import numpy as np
import pandas as pd
from PIL import Image
import json
import os
import torch
# from torchnet.meter import AUCMeter
import torchvision 
from tqdm import tqdm
from rand_aug import RandAugmentMC,RandAugmentwogeo

class CustomSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        # Copy attributes from the original dataset
        self.__dict__.update(dataset.__dict__)
            
def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class animal_dataset(Dataset): 
    def __init__(self, root_dir, transform, mode, pred=[], probability=[], saved=True,dataset='animal10n'): 
        
        self.transform = transform
        self.mode = mode  
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise
        saved = saved
        if self.mode=='test':
            data = torchvision.datasets.ImageFolder(root=f'{root_dir}/Animal10N/test', transform=None)
            self.test_label = []
            self.test_data = []
            if not saved:
                for i in tqdm(range(data.__len__()),disable=True):
                    image, label = data.__getitem__(i)
                    self.test_label.append(label)
                    self.test_data.append(image)
                torch.save(self.test_label,f'{root_dir}/Animal10N/test_label.pt')
                torch.save(self.test_data, f'{root_dir}/Animal10N/test_data.pt')
                print('data saved')
            else:
                self.test_data = torch.load(f'{root_dir}/Animal10N/test_data.pt')
                self.test_label = torch.load(f'{root_dir}/Animal10N/test_label.pt')
                self.test_data =  self.test_data    
                self.test_label = self.test_label           
        else:    
            train_data=[]
            train_label=[]
            data = torchvision.datasets.ImageFolder(root=f'{root_dir}/Animal10N/train', transform=None)
            if not saved:
                for i in tqdm(range(data.__len__()),disable=True):
                    image, label = data.__getitem__(i)
                    train_label.append(label)
                    train_data.append(image)
                noise_label = train_label
                torch.save(train_label, f'{root_dir}/Animal10N/train_label.pt')
                torch.save(train_data, f'{root_dir}/Animal10N/train_data.pt')
                print('data saved')
            else:
                train_data = torch.load(f'{root_dir}/Animal10N/train_data.pt')
                train_label = torch.load(f'{root_dir}/Animal10N/train_label.pt')
                train_data = train_data    
                train_label = train_label
                noise_label = train_label
            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
            elif self.mode =='meta':
                idx = list(range(50000))
                random.shuffle(idx)
                meta_id = idx[0:1000]
                self.train_data = [train_data[id] for id in meta_id]
                self.noise_label = [train_label[id] for id in meta_id]
            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]   
                    clean = (np.array(noise_label)==np.array(train_label))                                                               
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]     
                self.train_data = torch.utils.data.Subset(train_data, pred_idx)
                self.noise_label = torch.utils.data.Subset(noise_label, pred_idx)
    
     
                
    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            #img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2, target, prob            
        elif self.mode=='unlabeled':
            img = self.train_data[index]
            #img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2
        elif self.mode=='all':
            img, target = self.train_data[index], self.noise_label[index]
            #img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target, index    
            
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            #img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
            
        elif self.mode=='meta':
            img, target = self.train_data[index], self.noise_label[index]
            img = self.transform(img)            
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)         
        
        
class animal_dataloader():  
    def __init__(self, batch_size, num_workers, root_dir, saved=False,dataset='animal10n'):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.saved = saved
        self.dataset=dataset
        self.transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64),
            RandAugmentwogeo(n=1, m=10),

            transforms.ToTensor(),
            # transforms.Normalize((0.6959, 0.6537, 0.6371),
            #                      (0.3113, 0.3192, 0.3214)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

        ])
        self.transform_test = transforms.Compose([
                #transforms.Resize(32),
                transforms.ToTensor(),
                # transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

        ])    
     
    def run(self,mode,pred=[],prob=[]):
        if mode=='warmup':
            all_dataset = animal_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="all", saved=self.saved)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers, pin_memory=True)             
            return trainloader
                                     
        elif mode=='train':
            labeled_dataset = animal_dataset(dataset=self.dataset,   root_dir=self.root_dir, transform=self.transform_train, mode="labeled", pred=pred, probability=prob, saved=self.saved)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers, pin_memory=True)   
            
            unlabeled_dataset = animal_dataset(dataset=self.dataset,  root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled", pred=pred, saved=self.saved)                    
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers, pin_memory=True)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='test':
            test_dataset = animal_dataset(dataset=self.dataset,   root_dir=self.root_dir, transform=self.transform_test, mode='test', saved=self.saved)      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers, pin_memory=True)          
            return test_loader
        elif mode=='meta':
            meta_dataset = animal_dataset(dataset=self.dataset,  root_dir=self.root_dir, transform=self.transform_train, mode='meta', saved=self.saved)
            meta_loader = DataLoader(
                dataset=meta_dataset, 
                batch_size=200,
                shuffle=True,
                num_workers=self.num_workers, pin_memory=True)  
            return meta_loader 
            
        elif mode=='eval_train':
            eval_dataset = animal_dataset(dataset=self.dataset,  root_dir=self.root_dir, transform=self.transform_test, mode='all', saved=self.saved)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers, pin_memory=True)          
            return eval_loader        