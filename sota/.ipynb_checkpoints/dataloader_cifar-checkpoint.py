import io

from torch.utils.data import Dataset, DataLoader,Subset
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter
from torchvision.datasets.utils import download_url, check_integrity
import copy
from noise import *
from rand_aug import RandAugmentMC

def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class CustomSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        # Copy attributes from the original dataset
        self.__dict__.update(dataset.__dict__)


def split_meta_train(train_dset,num_meta,meta_goal,num_classes=10):
    if meta_goal!='clid':
        img_num_list = [int(num_meta/num_classes)] * num_classes
        data_list_val = {}
        for j in range(num_classes):
            data_list_val[j] = [i for i, label in enumerate(train_dset.train_labels) if label == j]
    
        idx_to_meta = []
        idx_to_train = []
        # print(img_num_list)
        for cls_idx, img_id_list in data_list_val.items():
            np.random.shuffle(img_id_list)
            img_num = img_num_list[int(cls_idx)]
            idx_to_meta.extend(img_id_list[:img_num])
            idx_to_train.extend(img_id_list[img_num:])
        train_dataset = CustomSubset(train_dset, idx_to_train)
        meta_dataset = CustomSubset(train_dset, idx_to_meta)
        return train_dataset, meta_dataset
    else:
        idx_to_meta = random.sample(range(len(train_dset)),num_meta)
        meta_dataset = data.Subset(train_dset, idx_to_meta)
        return train_dset, meta_dataset


class cifar_dataset(Dataset):

    def __init__(self,  dataset, noise_mode, r, root_dir, transform, mode, noise_file=None, pred=[], probability=[], log='', download=False):
        
        self.r = r # noise ratio
        self.transform = transform
        self.mode = mode  
        if dataset=='cifar10':
            self.base_folder = 'cifar-10-batches-py'
            self.url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
            self.filename = "cifar-10-python.tar.gz"
            self.tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
            self. train_list = [
                ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
                ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
                ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
                ['data_batch_4', '634d18415352ddfa80567beed471001a'],
                ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
            ]

            self.test_list = [
                ['test_batch', '40351d587109b95175f43aff81a1287e'],
            ]
        else:
            self.base_folder = 'cifar-100-python'
            self.url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
            self.filename = "cifar-100-python.tar.gz"
            self.tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
            self.train_list = [
                ['train', '16019d7e3df5f24257cddd939b257f8d'],
            ]

            self.test_list = [
                ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
            ]

        self.root = root_dir
        if download:
            self.download()
        if self.mode=='test':
            if dataset=='cifar10':                
                test_dic = unpickle('%s/cifar-10-batches-py/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['labels']
            elif dataset=='cifar100':
                test_dic = unpickle('%s/cifar-100-python/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['fine_labels']                            
        else:    
            train_data=[]
            train_label=[]
            if dataset=='cifar10':
                num_classes=10
                for n in range(1,6):
                    dpath = '%s/cifar-10-batches-py/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset=='cifar100':    
                num_classes =100
                train_dic = unpickle('%s/cifar-100-python/train'%root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))
            true_labels=copy.deepcopy(train_label)

            
            if noise_file:
                if os.path.exists(noise_file):
                    if noise_file.endswith('pt'):
                        #print(f'we are using purified labels from {noise_file}')
                        purified_labels = torch.load(noise_file)
                        noise_label = purified_labels.tolist()
                    elif noise_file.endswith('npy'):
                        with open(noise_file, 'rb') as f:
                            data = f.read()
                        file = io.BytesIO(data)
                        noise_label = np.load(file, allow_pickle=False).astype(np.int64)
                        noise_label = noise_label.tolist()
    
                    else:
                        noise_label = json.load(open(noise_file,"r"))
            else:    #inject noise 
                # if noise_mode == 'sym':
                #     C = uniform_mix_C(self.r, num_classes)
                #     # print(C)
                # elif noise_mode == 'flip':
                #     C = flip_labels_C(self.r, num_classes)
                # print(C)
                # noise_label = []
                # idx = list(range(50000))
                # random.shuffle(idx)
                # num_noise = int(self.r*50000)            
                # noise_idx = idx[:num_noise]
                # for i in range(50000):
                #     if i in noise_idx:
                #         if noise_mode=='sym':
                #             noiselabel = np.random.choice(num_classes, p=C[train_label[i]])
                #             noise_label.append(noiselabel)
                #         elif noise_mode=='asym':   
                #             noiselabel = self.transition[train_label[i]]
                #             noise_label.append(noiselabel)                    
                #     else:    
                #         noise_label.append(train_label[i])
                noise_label = get_noisy_label(dataset,train_label,num_classes,noise_mode,self.r,train_data,self.root,seed=0)
                        
                # print("save noisy labels to %s ..."%noise_file)        
                # json.dump(noise_label,open(noise_file,"w"))       
            
            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label

            elif self.mode == 'meta':
                idx = list(range(50000))
                random.shuffle(idx)
                meta_id = idx[0:1000]
                self.train_data = [train_data[id] for id in meta_id]
                self.noise_label = [train_label[id] for id in meta_id]
                # print(len(self.train_data), len(self.noise_label))

                # data_list_val = {}
                # for j in range(10):
                #     data_list_val[j] = [i for i, label in enumerate(self.noise_label) if label == j]
                #     # print("ratio class", j, ":", len(data_list_val[j]) / 1000 * 100)


            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]   
                    
                    clean = (np.array(noise_label)==np.array(train_label))                                                       
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(probability,clean)        
                    auc,_,_ = auc_meter.value()               
                    print('Numer of labeled samples:%d   AUC:%.3f\n'%(pred.sum(),auc))
                    log.flush()      
                    
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]                                               
                
                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]                          
                print("%s data has a size of %d"%(self.mode,len(self.noise_label)))            
                
    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2, target, prob            
        elif self.mode=='unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2
        elif self.mode=='all':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target, index        
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target
        elif self.mode=='meta':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target

           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            return
            # print('Files already downloaded and verified')
            # return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        
        
class cifar_dataloader():  
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        if self.dataset=='cifar10':
            # self.transform_train = transforms.Compose([
            #         transforms.RandomCrop(32, padding=4),
            #         transforms.RandomHorizontalFlip(),
            #         transforms.ToTensor(),
            #         transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
            #     ]) 
            self.transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32,
                                      padding=int(32 * 0.125),
                                      padding_mode='reflect'),
                RandAugmentMC(n=2, m=10),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),

            ])
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])    
        elif self.dataset=='cifar100':    
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]) 
            # self.transform_train = transforms.Compose([
            #     transforms.RandomHorizontalFlip(),
            #     transforms.RandomCrop(size=32,
            #                           padding=int(32 * 0.125),
            #                           padding_mode='reflect'),
            #     RandAugmentMC(n=2, m=10),
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),

            # ])
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])   
    def run(self,mode,pred=[],prob=[]):
        if mode=='warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="all",noise_file=self.noise_file)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
                                     
        elif mode=='train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="labeled", noise_file=self.noise_file, pred=pred, probability=prob,log=self.log)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled", noise_file=self.noise_file, pred=pred)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='all', noise_file=self.noise_file)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader        