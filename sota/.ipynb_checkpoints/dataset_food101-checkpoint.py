from __future__ import print_function
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
from PIL import Image
import PIL
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader, Subset
import random

# https://github.com/kuangliu/pytorch-retinanet/blob/master/transform.py
def resize(img, size, max_size=1000):
    '''Resize the input PIL image to the given size.
    Args:
      img: (PIL.Image) image to be resized.
      size: (tuple or int)
        - if is tuple, resize image to the size.
        - if is int, resize the shorter side to the size while maintaining the aspect ratio.
      max_size: (int) when size is int, limit the image longer size to max_size.
                This is essential to limit the usage of GPU memory.
    Returns:
      img: (PIL.Image) resized image.
    '''
    w, h = img.size
    if isinstance(size, int):
        size_min = min(w,h)
        sw = sh = float(size) / size_min
        
        ow = int(w * sw + 0.5)
        oh = int(h * sh + 0.5)
    else:
        ow, oh = size
        sw = float(ow) / w
        sh = float(oh) / h
    return img.resize((ow,oh), Image.BICUBIC)


class Food101N(Dataset):
    def __init__(self, split='train', data_path=None, transform=None):
        if data_path is None:
            data_path = 'image_list'

        if split == 'train':
    
            images = np.load(os.path.join(data_path, 'train_images.npy'))
            labels = np.load(os.path.join(data_path, 'train_targets.npy'))
            selected_indices= np.random.choice(len(images), size=55000, replace=False)
            self.image_list = images[selected_indices]
            self.targets  = labels[selected_indices]
            # self.image_list = np.load(os.path.join(data_path, 'train_images.npy'))
            # self.targets = np.load(os.path.join(data_path, 'train_targets.npy'))
        else:
            self.image_list = np.load(os.path.join(data_path, 'test_images.npy'))
            self.targets = np.load(os.path.join(data_path, 'test_targets.npy'))

        self.targets = self.targets - 1  # make sure the label is in the range [0, num_class - 1]
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = Image.open(image_path)
        # image = image.resize((256, 256), resample=PIL.Image.BICUBIC)
        image = resize(image, 256)

        if self.transform is not None:
            image = self.transform(image)

        label = self.targets[index]
        label = np.array(label).astype(np.int64)

        # return image, torch.from_numpy(label), index
        return image, label, index

    def __len__(self):
        return len(self.targets)

    def update_corrupted_label(self, noise_label):
        self.targets[:] = noise_label[:]

class CustomSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        # Copy attributes from the original dataset
        self.__dict__.update(dataset.__dict__)
        
def split_meta_train(train_dset,num_meta,meta_goal,num_classes=101):
    if meta_goal!='clid':
        img_num_list = [int(num_meta/num_classes)] * num_classes
        data_list_val = {}
        for j in range(num_classes):
            data_list_val[j] = [i for i, label in enumerate(train_dset.targets) if label == j]
    
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
        return meta_dataset
    else:
        idx_to_meta = random.sample(range(len(train_dset)),num_meta)
        meta_dataset = data.Subset(train_dset, idx_to_meta)
        return meta_dataset
