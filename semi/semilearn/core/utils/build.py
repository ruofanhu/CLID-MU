#Actu Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import math
import logging
import random
import torch
import torch.distributed as dist
from torch.nn import functional as F

from torch.utils.data import DataLoader,ConcatDataset
from semilearn.datasets.utils import get_collactor, name2sampler
from semilearn.nets.utils import param_groups_layer_decay, param_groups_weight_decay
import numpy as np
import copy
from semilearn.core.utils.noise import noisify_with_P, noisify_mnist_asymmetric, noisify_cifar10_asymmetric, noisify_cifar100_asymmetric, noisify_pairflip,noisify_instance,flip_labels_C_two



def manifold_loss_soft_exp_no1_(feat,logits,contrast_th):
# """ https://github.com/Hao-Ning/MEIDTM-Instance-Dependent-Label-Noise-Learning-with-Manifold-Regularized-Transition-Matrix-Estimatio/blob/master/run_ours.py """  
# they claim using the distance of feature space but using probability.    
# kernel: apply kernel trick or not
    temperature = contrast_th # temperature could affect
    probs_x_ulb = torch.softmax(logits, dim=-1)
    max_probs, y = torch.max(probs_x_ulb, dim=-1)
    feat = F.normalize(feat)
    normalized_probs = F.normalize(probs_x_ulb) #l2norm??
    
    m_feat = feat.shape[1] # m_feat is the # of dimensions of the features
    n_feat = feat.shape[0] # n_feat is the # of instances 
    

    sim_n = torch.mm(feat, feat.t())/temperature 
    # sim.fill_diagonal_(1)

    pos_mask = (sim_n>=0).float()
    # print(torch.sum((sim_n>=0).float()))
    sim_n = sim_n * pos_mask
    s_ij1 = F.softmax(sim_n,dim=-1)
    
    ####
    eps= 1e-7
    # weight_graph = torch.mm(max_probs.clone().reshape(-1,1), max_probs.clone().reshape(1,-1))
    weight_graph = torch.mm(normalized_probs,normalized_probs.t())
    
    weight_graph_ = weight_graph/weight_graph.sum(1,keepdim=True).detach()

    assert weight_graph.shape==(n_feat,n_feat)

    entr = torch.sum(-torch.log(s_ij1.detach()+eps)*weight_graph_,dim=1)
    # entr = torch.sum(-torch.log(s_ij1+eps)*weight_graph_,dim=1)

    return entr.mean()

def get_net_builder(net_name, from_name: bool):
    """
    built network according to network name
    return **class** of backbone network (not instance).

    Args
        net_name: 'WideResNet' or network names in torchvision.models
        from_name: If True, net_buidler takes models in torch.vision models. Then, net_conf is ignored.
    """
    if from_name:
        import torchvision.models as nets
        model_name_list = sorted(name for name in nets.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and callable(nets.__dict__[name]))

        if net_name not in model_name_list:
            assert Exception(f"[!] Networks\' Name is wrong, check net config, \
                               expected: {model_name_list}  \
                               received: {net_name}")
        else:
            return nets.__dict__[net_name]
    else:
        # TODO: fix bug here
        import semilearn.nets as nets
        builder = getattr(nets, net_name)
        
        return builder



def get_logger(name, save_path=None, level='INFO'):
    """
    create logger function
    """
    logger = logging.getLogger(name)
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s', level=getattr(logging, level))

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def get_dataset(args, algorithm, dataset, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=False):
    """
    create dataset

    Args
        args: argparse arguments
        algorithm: algrorithm name, used for specific return items in __getitem__ of datasets
        dataset: dataset name 
        num_labels: number of labeled data in dataset
        num_classes: number of classes
        seed: random seed
        data_dir: data folder
    """
    if algorithm =='fullysupervised_lossnet':
        args.ulb_num_labels=args.eval_set_size
    print('in getting dataset:',dataset,algorithm)
    from semilearn.datasets import get_eurosat, get_medmnist, get_semi_aves, get_cifar, get_svhn, get_stl10, get_imagenet, get_json_dset, get_pkl_dset, get_mnist,get_toy,get_clothing1m,get_json_data_weak

    if dataset == "eurosat":
        lb_dset, ulb_dset, eval_dset = get_eurosat(args, algorithm, dataset, num_labels, num_classes, data_dir=data_dir, include_lb_to_ulb=include_lb_to_ulb)
        test_dset = None
    elif dataset =="toy":
    
        lb_dset, ulb_dset, eval_dset = get_toy(args, algorithm, dataset, num_labels, num_classes, data_dir=data_dir, include_lb_to_ulb=False)
        test_dset = None        
    elif dataset in ["tissuemnist"]:
        lb_dset, ulb_dset, eval_dset = get_medmnist(args, algorithm, dataset, num_labels, num_classes, data_dir=data_dir,  include_lb_to_ulb=include_lb_to_ulb)
        test_dset = None
    elif dataset == "semi_aves":
        lb_dset, ulb_dset, eval_dset = get_semi_aves(args, algorithm, dataset, train_split='l_train_val', data_dir=data_dir)
        test_dset = None
    elif dataset == "semi_aves_out":
        lb_dset, ulb_dset, eval_dset = get_semi_aves(args, algorithm, "semi_aves", train_split='l_train_val', ulb_split='u_train_out', data_dir=data_dir)
        test_dset = None
    elif dataset in ["cifar10", "cifar100"]:
        lb_dset, ulb_dset, eval_dset = get_cifar(args, algorithm, dataset, num_labels, num_classes, data_dir=data_dir, include_lb_to_ulb=include_lb_to_ulb)
        test_dset = None
    elif dataset =='mnist':
        lb_dset, ulb_dset, eval_dset = get_mnist(args, algorithm, dataset, num_labels, num_classes, data_dir=data_dir, include_lb_to_ulb=include_lb_to_ulb)
        test_dset = None
    elif dataset == 'svhn':
        lb_dset, ulb_dset, eval_dset = get_svhn(args, algorithm, dataset, num_labels, num_classes, data_dir=data_dir, include_lb_to_ulb=include_lb_to_ulb)
        test_dset = None
    elif dataset == 'stl10':
        lb_dset, ulb_dset, eval_dset = get_stl10(args, algorithm, dataset, num_labels, num_classes, data_dir=data_dir, include_lb_to_ulb=include_lb_to_ulb)
        test_dset = None
    elif dataset == "imagenet":
        lb_dset, ulb_dset, eval_dset = get_imagenet(args, algorithm, dataset, num_labels, num_classes, data_dir=data_dir, include_lb_to_ulb=include_lb_to_ulb)
        test_dset = None
    elif dataset == "clothing1m":
        lb_dset, eval_dset = get_clothing1m(args, algorithm, dataset, num_labels, num_classes, data_dir=data_dir, include_lb_to_ulb=include_lb_to_ulb)
        ulb_dset = None
        test_dset = None
    # speech dataset
    elif dataset in ['esc50', 'fsdnoisy', 'gtzan', 'superbks', 'superbsi', 'urbansound8k']:
        lb_dset, ulb_dset, eval_dset, test_dset = get_pkl_dset(args, algorithm, dataset, num_labels, num_classes, data_dir=data_dir, include_lb_to_ulb=include_lb_to_ulb)
    elif dataset in ['aclImdb', 'ag_news', 'amazon_review', 'dbpedia', 'yahoo_answers', 'yelp_review']:
        lb_dset, ulb_dset, eval_dset, test_dset = get_json_dset(args, algorithm, dataset, num_labels, num_classes, data_dir=data_dir, include_lb_to_ulb=include_lb_to_ulb)
    elif dataset in ['agnews','yelp','imdb']:
        lb_dset, ulb_dset, dev_l_dest, dev_u_dset, test_dset = get_json_data_weak(args, algorithm, dataset, num_classes, data_dir=data_dir, include_lb_to_ulb=include_lb_to_ulb)
    else:
        raise NotImplementedError
    if dataset not in ['agnews','yelp','imdb']:
        y_train = np.array(lb_dset.targets)
        print('num_ytrain:',y_train.shape)
        noise_y_train = None
        keep_indices = None
        p = None
        if args.noise_ratio>0 or 'human' in args.noise_type and dataset !='clothing1m':
            noise_y_train = get_noisy_label(args,dataset,y_train,num_classes,lb_dset)
            lb_dset.targets = noise_y_train
    
        print('*'*50,'\n','Actual noise ratio:',np.mean(lb_dset.targets!=np.array(lb_dset.true_labels)),'\n','*'*50)
                
        if args.meta_goal:
            meta_dset = get_meta_set(args,ulb_dset,lb_dset,num_classes)
            print('label_set:',len(lb_dset),'meta_set:',len(meta_dset))
        else:
            meta_dset = None
        
        # dataset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset, 'eval': eval_dset, 'meta': meta_dset,'test':eval_dset} #meta_dset
        if test_dset is None:
            dataset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset, 'eval': meta_dset, 'meta': meta_dset,'test':eval_dset} #meta_dset
        else:
            dataset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset, 'eval': meta_dset, 'meta': meta_dset,'test':test_dset} #meta_dset            
    else:
        print('*'*50,'\n','Actual noise ratio:',np.mean(lb_dset.targets!=np.array(lb_dset.true_labels)),'\n','*'*50)

        meta_dset = get_meta_set_text(args,dev_u_dset,dev_l_dest)
        dataset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset, 'eval': meta_dset, 'meta': meta_dset,'test':test_dset} #meta_dset
        keep_indices =None
        

        
    args.log = {'true_label':lb_dset.true_labels, 'train_labels':lb_dset.targets,'clean_indices':keep_indices,'iteration':[],'sample_idx': [] ,\
            'w_ce_loss':[], 'w_lid_logits_l2':[], 'w_lid_feat_l2':[],'w_lid_logits_cos':[], 'w_lid_feat_cos':[],'lid_feat_l2':[],\
            'lid_feat_cos':[],'lid_logits_l2':[],'lid_logits_cos':[],'unl_ce_loss':[],'labi_loss':[],'labi_lid_logits_l2':[],'labi_lid_feat_l2':[],\
             'labi_lid_logits_cos':[],'labi_lid_feat_cos':[],'feat_std':[],'cov_loss':[],'total_meta_loss':[],'auc':[],'f1':[],'auprc':[]
               }

    return dataset_dict

def get_noisy_label(args,dataset,y_train,num_classes,lb_dset):
    if 'human' in args.noise_type:
        lbs_file =f'lb_num_{args.num_labels}_{args.noise_type}_seed_{args.seed}.pt'
    else:
        lbs_file =f'lb_num_{args.num_labels}_{args.noise_type}_{args.noise_ratio}_seed_{args.seed}.pt'
    
    lbs_file_path = os.path.join(args.data_dir,args.dataset,'noisy_label',lbs_file)
    # if os.path.exists(lbs_file_path):
    #     noise_y_train = torch.load(lbs_file_path)
    #     return noise_y_train

    if args.noise_type=='sym':
        
        noise_y_train, p, keep_indices = noisify_with_P(y_train, nb_classes=num_classes, noise=args.noise_ratio, random_state=args.seed)
        if len(keep_indices)==0:
            ids=random.sample(range(len(y_train)), round(len(y_train)*args.noise_ratio)) 
            noise_y_train = np.array(lb_dset.targets)
            for i in ids:
                noise_y_train[i] = 1-noise_y_train[i]
    elif args.noise_type=='asy':
        if dataset == 'cifar10':
            noise_y_train, p, keep_indices = noisify_cifar10_asymmetric(y_train, noise=args.noise_ratio, random_state=args.seed)
        elif dataset == 'cifar100':
            noise_y_train, p, keep_indices = noisify_cifar100_asymmetric(y_train, noise=args.noise_ratio, random_state=args.seed)
        elif dataset == 'mnist':
            noise_y_train, p, keep_indices = noisify_mnist_asymmetric(y_train, noise=args.noise_ratio, random_state=args.seed)
            
    elif args.noise_type=='flip2':
        noise_y_train, p, keep_indices = flip_labels_C_two(y_train, noise=args.noise_ratio, nb_classes=num_classes,random_state=args.seed)
        
    elif args.noise_type=='ins':
        noise_y_train = noisify_instance(lb_dset.data,y_train,num_classes,noise_rate=args.noise_ratio,random_state=args.seed )
                       
    elif 'human' in args.noise_type:
        if dataset == 'cifar10':
            noise_file_name='CIFAR-10_human.pt'
        elif dataset == 'cifar100':
            noise_file_name='CIFAR-100_human.pt'
            
        noise_path = os.path.join(args.data_dir,args.dataset,noise_file_name)
        noise_file= torch.load(noise_path)
        lb_dump_path = os.path.join(args.data_dir,args.dataset,'labeled_idx', f'lb_labels{args.num_labels}_seed{args.seed}_idx.npy')
        # lb_ids=np.array([e['idx_lb'] for e in lb_dset])
        lb_ids = np.load(lb_dump_path)
        # lb_idx =np.array([e['idx_lb'] for e in lb_dset])
        # print('check lb ids:', np.mean(lb_ids==lb_idx))
        if dataset == 'cifar10':
            if 'worst' in args.noise_type:
                noisy_label = noise_file['worse_label']
            elif 'aggre' in args.noise_type:
                noisy_label = noise_file['aggre_label']
            elif 'random' in args.noise_type:
                noisy_label = noise_file['random_label1']
            
        elif dataset == 'cifar100':
            noisy_label = noise_file['noisy_label']
            
        # if lb_ids.shape[0]==50000:
        #     return noisy_label     
        noise_y_train = noisy_label[lb_ids]

        
    # torch.save(noise_y_train,lbs_file_path)
    print('already save noisy label file.')
    return noise_y_train


def get_meta_set_text(args,ulb_dset,lb_dset):
    # meta_file = f'meta_num_{args.num_labels}_{args.noise_type}_ratio_{args.noise_ratio}_{args.eval_set_size}_seed_{args.seed}.pt'
    # meta_file_path = os.path.join(args.data_dir,args.dataset,'meta_id',meta_file)
            
    if 'feat_expno1N' in args.meta_goal:
        if 'fully' in args.algorithm:
            dset = ConcatDataset([ulb_dset, lb_dset])
      

        N=len(dset)
        if args.eval_set_size>N:
            args.eval_set_size=N
        indices_eval = random.sample(range(N), args.eval_set_size) #eval set is the one for cal_meta
        random.shuffle(indices_eval)
        meta_dset = torch.utils.data.Subset(dset,indices_eval) # true evaluation set for cal_meta_gradient
        # meta_y=np.array([ulb_dset.targets[i] for i in indices_eval]) #we did not actually use the label
        
    elif 'feat_expno1N' not in args.meta_goal:        
    
        if args.meta_goal=='cer':
            lb_dset = ConcatDataset([ulb_dset, lb_dset])
        #     lbs = lb_dset.true_labels
        N=len(lb_dset)
        if args.eval_set_size>N:
            print('the request evaluation set is larger than available,number of available:',N)
            args.eval_set_size=N
        indices_eval = random.sample(range(N), args.eval_set_size) #eval set is the one for cal_meta
        random.shuffle(indices_eval)
        
        meta_dset = torch.utils.data.Subset(lb_dset,indices_eval) # true evaluation set for cal_meta_gradient
    return meta_dset


def get_meta_set(args,ulb_dset,lb_dset,num_classes):
    # meta_file = f'meta_num_{args.num_labels}_{args.noise_type}_ratio_{args.noise_ratio}_{args.eval_set_size}_seed_{args.seed}.pt'
    # meta_file_path = os.path.join(args.data_dir,args.dataset,'meta_id',meta_file)
            
    if 'feat_expno1N' in args.meta_goal:
        if 'fully' in args.algorithm:
            dset = lb_dset
        else:
            dset = ulb_dset        

        N=len(dset)
        if args.eval_set_size>N:
            args.eval_set_size=N
        indices_eval = random.sample(range(N), args.eval_set_size) #eval set is the one for cal_meta
        random.shuffle(indices_eval)
        meta_dset = torch.utils.data.Subset(dset,indices_eval) # true evaluation set for cal_meta_gradient
        # meta_y=np.array([ulb_dset.targets[i] for i in indices_eval]) #we did not actually use the label
        
    elif 'feat_expno1N' not in args.meta_goal:        
        N=len(ulb_dset)
        if args.eval_set_size>N:
            print('the request evaluation set is larger than available')
            args.eval_set_size=N
            
        num_c = args.eval_set_size//num_classes        
        indices_eval = []
        lbs = lb_dset.targets
        if args.meta_goal=='cer':
            lbs = lb_dset.true_labels
        for c in range(num_classes):
            data_class = [i for i, label in enumerate(lbs) if label == c]
            indices_eval.extend(random.sample(data_class,num_c))
        random.shuffle(indices_eval)
        # torch.save(indices_eval,meta_file_path)
        
        meta_dset = torch.utils.data.Subset(lb_dset,indices_eval) # true evaluation set for cal_meta_gradient
        # meta_y=np.array([lb_dset.targets[i] for i in indices_eval])
    return meta_dset
    
def get_data_loader(args,
                    dset,
                    batch_size=None,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=False,
                    data_sampler='RandomSampler',
                    num_epochs=None,
                    num_iters=None,
                    generator=None,
                    drop_last=True,
                    distributed=False):
    """
    get_data_loader returns torch.utils.data.DataLoader for a Dataset.
    All arguments are comparable with those of pytorch DataLoader.
    However, if distributed, DistributedProxySampler, which is a wrapper of data_sampler, is used.
    
    Args
        num_epochs: total batch -> (# of batches in dset) * num_epochs 
        num_iters: total batch -> num_iters
    """

    assert batch_size is not None
    if num_epochs is None:
        num_epochs = args.epoch
    if num_iters is None:
        num_iters = args.num_train_iter
        
    collact_fn = get_collactor(args, args.net)

    if data_sampler is None:
        return DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collact_fn,
                          num_workers=num_workers, drop_last=drop_last, pin_memory=pin_memory)

    if isinstance(data_sampler, str):
        data_sampler = name2sampler[data_sampler]

        if distributed:
            assert dist.is_available()
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
        else:
            num_replicas = 1
            rank = 0

        per_epoch_steps = num_iters // num_epochs

        num_samples = per_epoch_steps * batch_size * num_replicas

        return DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collact_fn,
                          pin_memory=pin_memory, sampler=data_sampler(dset, num_replicas=num_replicas, rank=rank, num_samples=num_samples),
                          generator=generator, drop_last=drop_last)

    elif isinstance(data_sampler, torch.utils.data.Sampler):
        return DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                          collate_fn=collact_fn, pin_memory=pin_memory, sampler=data_sampler,
                          generator=generator, drop_last=drop_last)

    else:
        raise Exception(f"unknown data sampler {data_sampler}.")


def get_optimizer(net, optim_name='SGD', lr=0.1, momentum=0.9, weight_decay=0, layer_decay=1.0, nesterov=True, bn_wd_skip=True):
    '''
    return optimizer (name) in torch.optim.
    If bn_wd_skip, the optimizer does not apply
    weight decay regularization on parameters in batch normalization.
    '''
    assert layer_decay <= 1.0

    no_decay = {}
    if hasattr(net, 'no_weight_decay') and bn_wd_skip:
        no_decay = net.no_weight_decay()
    
    if layer_decay != 1.0:
        per_param_args = param_groups_layer_decay(net, lr, weight_decay, no_weight_decay_list=no_decay, layer_decay=layer_decay)
    else:
        per_param_args = param_groups_weight_decay(net, weight_decay, no_weight_decay_list=no_decay)

    if optim_name == 'SGD':
        optimizer = torch.optim.SGD(per_param_args, lr=lr, momentum=momentum, weight_decay=weight_decay,
                                    nesterov=nesterov)
    elif optim_name == 'AdamW':
        optimizer = torch.optim.AdamW(per_param_args, lr=lr, weight_decay=weight_decay)
    
    return optimizer


def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
    '''
    Get cosine scheduler (LambdaLR).
    if warmup is needed, set num_warmup_steps (int) > 0.
    '''
    from torch.optim.lr_scheduler import LambdaLR
    def _lr_lambda(current_step):
        '''
        _lr_lambda returns a multiplicative factor given an interger parameter epochs.
        Decaying criteria: last_epoch
        '''

        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def get_step_schedule(optimizer,step_size=4,gamma=0.1):
    '''
    Get step scheduler (StepLR).
    '''
    from torch.optim.lr_scheduler import StepLR


    return StepLR(optimizer, step_size, gamma)

def adjust_learning_rate(optimizer, iters):

    lr = args.lr * ((0.1 ** int(iters >= 16200)) * (0.1 ** int(iters >= 18000)))  # For WRN-28-10
    #lr = args.lr * ((0.1 ** int(iters >= 20000)) * (0.1 ** int(iters >= 25000)))  # For ResNet32
    # log to TensorBoard
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_port():
    """
    find a free port to used for distributed learning
    """
    pscmd = "netstat -ntl |grep -v Active| grep -v Proto|awk '{print $4}'|awk -F: '{print $NF}'"
    procs = os.popen(pscmd).read()
    procarr = procs.split("\n")
    tt= random.randint(15000, 30000)
    if tt not in procarr:
        return tt
    else:
        return get_port()
