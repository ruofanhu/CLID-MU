from PIL import Image
import os
import os.path
import numpy as np
import sys
import pickle
import torch
import torch.utils.data as data
import tools
from torchvision.datasets.utils import download_url, check_integrity
import torchvision.transforms as transforms
import copy
import random
from torch.utils.data import Dataset, DataLoader, Subset
import json
from collections import defaultdict

def make_imb_data(max_num, class_num, gamma):
    mu = np.power(1 / gamma, 1 / (class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / gamma))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))
    # print(class_num_list)
    return list(class_num_list)
    
def select_data_by_class_count(num_instances_per_class, data, class_labels):
    # Validate inputs
    if len(num_instances_per_class) == 0 or len(data) != len(class_labels):
        raise ValueError("Input lists are invalid or of mismatched lengths.")

    # Initialize counters for each class
    selected_data = []
    selected_labels = []
    class_counters = defaultdict(int)  # Keeps track of selected instances per class

    # Iterate over the data and labels
    for item, label in zip(data, class_labels):
        # Check if there are remaining slots for this class
        if class_counters[label] < num_instances_per_class[label]:
            selected_data.append(item)
            selected_labels.append(label)
            class_counters[label] += 1

        # Stop early if all classes have their required instances
        if all(class_counters[cls] >= num_instances_per_class[cls] for cls in range(len(num_instances_per_class))):
            break

    return selected_data, selected_labels   


def uniform_mix_C(mixing_ratio, num_classes):
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
        (1 - mixing_ratio) * np.eye(num_classes)


def flip_labels_C(corruption_prob, num_classes, seed=1):
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    return C


def flip_labels_C_two(corruption_prob, num_classes, seed=1):
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i], 2, replace=False)] = corruption_prob / 2
    return C
    
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

def get_imbalanced_dataset(train_dset,max_num, num_classes, gamma):
    num_instances_per_class = make_imb_data(max_num, num_classes, gamma)
    idx_to_train = []
    for j in range(num_classes):
        id_j= [i for i, label in enumerate(train_dset.train_true_labels) if label == j][:num_instances_per_class[j]]
        idx_to_train.extend(id_j)
    np.random.shuffle(idx_to_train)
    train_dataset = CustomSubset(train_dset, idx_to_train)
    return train_dataset
    
class CustomSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        # Copy attributes from the original dataset
        self.__dict__.update(dataset.__dict__)
        
class CIFAR10(data.Dataset):
    base_folder = 'cifar-10-batches-py'
    url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, device, root='', train=True, meta=True, num_meta=1000,
                 corruption_prob=0, corruption_type='unif', transform=None, target_transform=None,
                 download=False, seed=1, strong_t=None, normalize = None):
        self.count = 0
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.meta = meta
        self.corruption_prob = corruption_prob
        self.num_meta = num_meta
        self.strong_transform = strong_t
        self.normalize = normalize#transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))

        if self.strong_transform:
            print('Strong transform...')
        

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            self.train_coarse_labels = []

            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                    # img_num_list = [int(self.num_meta/10)] * 10
                    num_classes = 10
                    max_num=5000

                else:
                    self.train_labels += entry['fine_labels']
                    self.train_coarse_labels += entry['coarse_labels']
                    # img_num_list = [int(self.num_meta/100)] * 100
                    num_classes = 100
                    max_num=500

                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))   # convert to HWC
            
            self.train_true_labels = copy.deepcopy(self.train_labels)
          
            if corruption_type == 'unif':
                C = uniform_mix_C(self.corruption_prob, num_classes)
                # print(C)
                self.C = C
            elif corruption_type == 'flip':
                C = flip_labels_C(self.corruption_prob, num_classes)
                # print(C)
                self.C = C

            elif corruption_type == 'inst':
                data_ = torch.from_numpy(self.train_data).reshape((-1, 3072)).float()
                targets_ = torch.from_numpy(np.array(self.train_labels))
                dataset = zip(data_, targets_)
                self.train_labels = tools.get_instance_noisy_label(device, self.corruption_prob, dataset, targets_,
                                                                   num_classes, 3072,
                                                                   0.1, seed)
            elif corruption_type == 'flip2':
                C = flip_labels_C_two(self.corruption_prob, num_classes)
                # print(C)
                self.C = C
            elif corruption_type == 'hierarchical':
                assert num_classes == 100, 'You must use CIFAR-100 with the hierarchical corruption.'
                coarse_fine = []
                for i in range(20):
                    coarse_fine.append(set())
                for i in range(len(self.train_labels)):
                    coarse_fine[self.train_coarse_labels[i]].add(self.train_labels[i])
                for i in range(20):
                    coarse_fine[i] = list(coarse_fine[i])

                C = np.eye(num_classes) * (1 - corruption_prob)

                for i in range(20):
                    tmp = np.copy(coarse_fine[i])
                    for j in range(len(tmp)):
                        tmp2 = np.delete(np.copy(tmp), j)
                        C[tmp[j], tmp2] += corruption_prob * 1/len(tmp2)
                self.C = C
                # print(C)

            # else:
                # assert False, "Invalid corruption type '{}' given. Must be in {'unif', 'flip', 'hierarchical', 'inst'}".format(corruption_type)
            np.random.seed(seed)
            if corruption_type == 'clabels':
                mean = [x / 255 for x in [125.3, 123.0, 113.9]]
                std = [x / 255 for x in [63.0, 62.1, 66.7]]

                test_transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize(mean, std)])

                # obtain sampling probabilities
                sampling_probs = []
                print('Starting labeling')

                sampling_probs = np.concatenate(sampling_probs, 0)
                print('Finished labeling 1')

                new_labeling_correct = 0
                argmax_labeling_correct = 0
                for i in range(len(self.train_labels)):
                    old_label = self.train_labels[i]
                    new_label = np.random.choice(num_classes, p=sampling_probs[i])
                    self.train_labels[i] = new_label
                    if old_label == new_label:
                        new_labeling_correct += 1
                    if old_label == np.argmax(sampling_probs[i]):
                        argmax_labeling_correct += 1
                print('Finished labeling 2')
                print('New labeling accuracy:', new_labeling_correct / len(self.train_labels))
                print('Argmax labeling accuracy:', argmax_labeling_correct / len(self.train_labels))

            elif corruption_type == 'inst':
                print('instance noise!')
            elif corruption_type.startswith('human'):
                if num_classes ==100:
                    noise_file=torch.load(f'{self.root}/cifar100/CIFAR-100_human.pt')
                    self.train_labels = noise_file['noisy_label']
                elif num_classes ==10:
                    noise_file=torch.load(f'{self.root}/cifar10/CIFAR-10_human.pt')
                    if 'worst' in corruption_type:
                        self.train_labels = noise_file['worse_label']
                    elif 'aggre' in corruption_type:
                        self.train_labels = noise_file['aggre_label']
                    elif 'random' in corruption_type:
                        self.train_labels = noise_file['random_label1']
            else:
                for i in range(len(self.train_labels)):
                    self.train_labels[i] = np.random.choice(num_classes, p=self.C[self.train_labels[i]])
                # print('train', len(self.train_labels))
                # print('type', type(self.train_labels))
                self.corruption_matrix = self.C
                
        if not self.train:
            f = self.test_list[0][0]
            file = os.path.join(root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    
    def __getitem__(self, index):
        if self.train:
            img, target, true_label = self.train_data[index], self.train_labels[index], self.train_true_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)
        
        if self.transform is not None:
            if self.strong_transform:
                img_w = self.strong_transform(img)
                # img_w = self.normalize(img_w)
            else:
                img_w = self.transform(img)
        # if self.transform is not None:
        #     img_w = self.transform(img)
        if self.strong_transform:
            img_s1 = self.strong_transform(img)
            # img_s1 = self.normalize(img_s1)
            
            img_s2 = self.strong_transform(img)        
            # img_s2 = self.normalize(img_s2)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.train:
            return img_w, img_s1,img_s2,target, index, true_label
        else:
            return img_w, target, index
       
        
    def __len__(self):
        if self.train:
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
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)


class CIFAR100(CIFAR10):
    base_folder = 'cifar-100-python'
    url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]


