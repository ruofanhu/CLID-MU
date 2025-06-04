from meta_net import vri, vri_dec, vri_prior, VNet
from PreResNet import ResNet18
from resnet32 import ResNet32
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from dataloader import CIFAR10, CIFAR100, split_meta_train
import argparse
import os, warnings
from rand_aug import RandAugmentMC
import numpy as np
from utilities import *
from tqdm import tqdm
import datetime
from torch.nn import functional as F
import logging
from wideresnet import WideResNet

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--corruption_prob', type=float, default=0.4, help='label noise')
parser.add_argument('--corruption_type', '-ctype', type=str, default='unif', help='Type of corruption ("unif" or "flip" or "inst").')
parser.add_argument('--num_meta', type=int, default=1000)
parser.add_argument('--warmup_epochs', default=10, type=int, help='number of total epochs for warm up')
parser.add_argument('--epochs', default=150, type=int, help='number of total epochs to run')
parser.add_argument('--iters', default=60000, type=int, help='number of total iters to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=100, type=int, help='mini-batch size (default: 100)')
parser.add_argument('--meta_bsz', default=100, type=int, help='meta mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--meta_lr', default=3e-4, type=float, help='learning rate for the meta net')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay (default: 5e-0.4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--layers', default=28, type=int, help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=10, type=int, help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0, type=float, help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false', help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='PreResNet-18', type=str, help='name of experiment')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--prefetch', type=int, default=12, help='Pre-fetching threads.')
parser.add_argument('--var', type=float, default=0.1, help='Pre-fetching threads.')
parser.add_argument('--gpuid', type=str, default='0')
parser.add_argument('--lam', type=float, default=0.01)
# parser.add_argument('--alpha', default=0.2, type=float, help='parameter for Beta in mixup')
parser.add_argument('--tau', default=0.1, type=float, help='tau for clid loss')
parser.add_argument('--meta_goal', type=str, default='ce',help='ce,ce_sloss, mae_sloss,clid')
parser.add_argument('--scheduler', type=str, default='m_step',help='cos, m_step')

parser.add_argument('--eval_iter', default=250, type=int, help='evaluation every n iterations')
parser.add_argument('--data_root', type=str, default='/home/rhu/r_work/higher_semilearn/data',help='data folder')
parser.add_argument('--Tmax', type=int, default=10,help='cosine period')

parser.set_defaults(augment=True)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")



def build_dataset(root, args):
    if args.dataset == 'cifar10':
        normalize = transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
    elif args.dataset == 'cifar100':
        normalize = transforms.Normalize((0.507,0.487,0.441),(0.267,0.265,0.276))
    if args.augment:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),  (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    strong_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32 * 0.125),
                              padding_mode='reflect'),
        RandAugmentMC(n=2, m=10),
        transforms.ToTensor(),
        normalize
    ])

    if args.dataset == 'cifar10':

        # train_data_meta = CIFAR10(device,
        #         root=root, train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
        #         corruption_type=args.corruption_type, transform=train_transform, download=True,
        #                           strong_t=None, normalize=normalize)

        train_data = CIFAR10(device,
                root=root, train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
                corruption_type=args.corruption_type, transform=strong_transform, download=True,
                             seed=args.seed, strong_t=strong_transform, normalize=normalize)

        test_data = CIFAR10(device,root=root, train=False, transform=test_transform, download=True, normalize=normalize)

        train_data, train_data_meta = split_meta_train(train_data,args.num_meta,args.meta_goal,num_classes=10)

    elif args.dataset == 'cifar100':
        # train_data_meta = CIFAR100(device,
        #     root=root, train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
        #     corruption_type=args.corruption_type, transform=train_transform, download=True, strong_t=None, normalize=normalize)
        train_data = CIFAR100(device,
            root=root, train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=strong_transform, download=True, seed=args.seed,
                              strong_t=strong_transform, normalize=normalize)
        test_data = CIFAR100(device,root=root, train=False, transform=test_transform, download=True, normalize=normalize)
        train_data, train_data_meta = split_meta_train(train_data,args.num_meta,args.meta_goal,num_classes=100)

    train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True,
            num_workers=args.prefetch, pin_memory=True)
    train_meta_loader = torch.utils.data.DataLoader(
            train_data_meta, batch_size=args.meta_bsz, shuffle=True,
            num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.prefetch, pin_memory=True)

    return train_loader, train_meta_loader, test_loader

def build_classifier(args):
    # if args.corruption_type =='unif' or  args.corruption_type =='inst':
    cnn = WideResNet(args.layers, args.dataset == 'cifar10' and 10 or 100,
                   args.widen_factor)
    # else:
    #     cnn = ResNet32(args.dataset == 'cifar10' and 10 or 100)
    # cnn = ResNet18(args.dataset == 'cifar10' and 10 or 100)

    if torch.cuda.is_available():
        cnn.cuda()
        torch.backends.cudnn.benchmark = True

    return cnn

def mixup(net, data_loader, optimizer, alpha, num_classes):

    net.train()
    num_iter = (len(data_loader.dataset) // data_loader.batch_size) + 1
    losses = 0.0

    for batch_idx, (inputs, labels, indices, true_labels) in enumerate(data_loader):#(tqdm.tqdm(data_loader, ncols=0)):
        l = np.random.beta(alpha, alpha)
        labels = torch.nn.functional.one_hot(labels.long(), num_classes).float()
        inputs, labels = inputs.cuda(), labels.cuda()

        idx = torch.randperm(inputs.size(0))

        input_a, input_b = inputs, inputs[idx]
        target_a, target_b = labels, labels[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        logits, _ = net(mixed_input)
        loss = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss

    return losses/num_iter



def warm_up(model, warm_loader, optimizer):
    model.train()
    acc_train = 0.0
    train_loss = 0.0
    for batch_idx, (inputs,_,_, targets, index,_) in enumerate(tqdm(warm_loader,disable=True)):
        num = batch_idx
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs, _ = model(inputs)
        prec_train = accuracy(outputs.data, targets.data, topk=(1,))[0]
        acc_train += prec_train
        loss = F.cross_entropy(outputs, targets.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        return train_loss/(num+1), acc_train/(num+1)


def train(train_loader, train_meta_loader, model, vnet, optimizer_model, optimizer_vnet, epoch,args):
    print('Epoch: %d, lr: %.5f' % (epoch, optimizer_model.param_groups[0]['lr']))

    train_loss = 0
    meta_loss = 0
    acc_meta = 0.0
    acc_train = 0.0

    num = 0
    criterion = nn.CrossEntropyLoss().cuda()

    train_meta_loader_iter = iter(train_meta_loader)
    for batch_idx, (inputs,_,_, targets, indices,true_labels) in enumerate(tqdm(train_loader, ncols=0,disable=True)):
        num = batch_idx
        model.train()
        meta_model = build_classifier(args).cuda()
        meta_model.load_state_dict(model.state_dict())

        oringal_targets = targets.cuda()
        inputs, targets = inputs.cuda(), targets.cuda()
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes).float().cuda()

        # ========================== step 1 ====================================
        outputs, feat = meta_model(inputs)
        # dict0 = {'feature':feat.detach(), 'target':targets}
        # mean, log_var, v_lambda = vnet(dict0)
        # dict1 = {'feature':feat.detach()}
        # mean_p, log_var_p, _ = vnet(dict1)
        cost = F.cross_entropy(outputs, targets, reduce=False)
        cost_v = torch.reshape(cost, (len(cost), 1))
        v_lambda = vnet(cost_v.data)
        ##########
        norm_c = torch.sum(v_lambda)
        if norm_c != 0:
            v_lambda_norm = v_lambda / norm_c
        else:
            v_lambda_norm = v_lambda
        l_f_meta = torch.sum(cost * torch.squeeze(v_lambda_norm)) 
        # updata copy_model`s params
        meta_model.zero_grad()
        grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
        meta_lr = optimizer_model.param_groups[0]['lr']
        meta_model.update_params(lr_inner=meta_lr, source_params=grads)
        del grads

        # ========================= step 2 =====================================

        try:
            inputs_val, _,_,targets_val, _, targets_val_true = next(train_meta_loader_iter)
        except:
            train_meta_loader_iter = iter(train_meta_loader)
            inputs_val, _,_, targets_val, _, targets_val_true = next(train_meta_loader_iter)
        inputs_val, targets_val,targets_val_true = inputs_val.cuda(), targets_val.cuda(),targets_val_true.cuda()  # [500,3,32,32], [500]
        
        y_g_hat, feat_val = meta_model(inputs_val)
        prec_train = accuracy(y_g_hat.data, targets_val.data, topk=(1,))[0]
        acc_meta += prec_train
        if args.meta_goal=='ce':
            l_g_meta = F.cross_entropy(y_g_hat,targets_val_true.long())
        elif args.meta_goal=='ce_sloss' or args.meta_goal=='ce_noisy':
            l_g_meta = F.cross_entropy(y_g_hat,targets_val.long())
        elif args.meta_goal=='mae' or args.meta_goal=='mae_sloss':
            l_g_meta = mae_loss(y_g_hat, targets_val.long())
        ### Apply clid_loss
        elif args.meta_goal=='clid':
            l_g_meta = clid_loss(feat_val,y_g_hat,args.tau)

        # update vnet params
        optimizer_vnet.zero_grad()
        l_g_meta.backward()
        optimizer_vnet.step()

        # ========================= step 3 =====================================
        outputs, feat = model(inputs)
        prec_train = accuracy(outputs.data, oringal_targets.data, topk=(1,))[0]
        acc_train += prec_train

        cost = F.cross_entropy(outputs, targets, reduce=False)
        cost_v = torch.reshape(cost, (len(cost), 1))
        with torch.no_grad():
            v_lambda = vnet(cost_v.data)
        ##########
        norm_c = torch.sum(v_lambda)
        if norm_c != 0:
            v_lambda_norm = v_lambda / norm_c
        else:
            v_lambda_norm = v_lambda
        loss = torch.sum(cost * torch.squeeze(v_lambda_norm))

        # loss = loss_function(w_new*outputs, targets_onehot)

        # update model params
        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        train_loss += loss.item()
        meta_loss += l_g_meta.item()

    return train_loss/(num+1), meta_loss/(num+1), acc_train/(num+1), acc_meta/(num+1)

def build_training(args):
    model = build_classifier(args).cuda()
    # vnet = vri(576, 1024, 512, num_classes).cuda()
    vnet = VNet(1, 100, 1).cuda()

    optimizer_model = torch.optim.SGD(model.params(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.scheduler!='cos':
        sch_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer_model, milestones=[36, 38], gamma=0.1)
    else:
        sch_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_model, T_max=args.Tmax, eta_min=1e-3)    
    optimizer_vnet = torch.optim.SGD(vnet.params(), args.meta_lr, weight_decay=args.weight_decay)

    return model, vnet, \
           optimizer_model, \
           sch_lr, optimizer_vnet

def test_ensembel(model_1, model_2, test_loader):
    model_1.eval()
    model_2.eval()
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs_1, _ = model_1(inputs)
            outputs_2, _ = model_2(inputs)
            outputs = (outputs_1 + outputs_2) / 2
            test_loss += F.cross_entropy(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    return accuracy

def main(args,mytxt):
    torch.manual_seed(args.seed)

    root = args.data_root
    train_loader_2, train_meta_loader_2, test_loader = build_dataset(root, args)

    global  num_classes
    if args.dataset == 'cifar10':
        num_classes = 10
        args.epochs = 100
        args.warmup_epochs=10
    else:
        num_classes = 100
        args.epochs =100
        args.warmup_epochs=30

    model_2, vnet_2, optimizer_model_2, sch_lr_2, optimizer_vnet_2 \
        = build_training(args)

    best_acc, meta_loss, acc_meta = 0.0, 0.0, 0.0
    best_checkpoints = {}
    best_checkpoints_accu = {}
    ## warm up
    if args.meta_goal=='ce_sloss' or args.meta_goal=='mae_sloss':
        for epoch in range(args.warmup_epochs):
            train_loss,train_acc = warm_up(model_2, train_loader_2, optimizer_model_2)
            print(f'Warmup: train_loss:{train_loss}, train_acc:{train_acc}')
            print(f'Warmup: train_loss:{train_loss}, train_acc:{train_acc}',file=mytxt)

        # args.epochs =args.epochs-args.warmup_epochs
        noise_ratio, meta_set = eval_train(model_2, train_loader_2)
        train_meta_loader_2 = torch.utils.data.DataLoader(
                meta_set, batch_size=args.meta_bsz, shuffle=True,
                num_workers=args.prefetch, pin_memory=True)
        print(f'Meta_set_noise:{noise_ratio}')
        print(f'Meta_set_noise:{noise_ratio}',file=mytxt)

    for epoch in range(args.epochs):
        train_loss, meta_loss, acc_train, acc_meta = train(train_loader_2, train_meta_loader_2, model_2,
                                                                      vnet_2,optimizer_model_2, optimizer_vnet_2, epoch,args)
        
        meta_acc,meta_clid,sve, lid,lsvr,cov = test_meta(model=model_2, meta_loader=train_meta_loader_2,meta_goal=args.meta_goal,args=args)
        test_acc, test_outputs,targets = test(model=model_2, test_loader=test_loader)
        
        meta_clid_results,best_checkpoints = update_best_results_dict(best_checkpoints,meta_clid, test_outputs,epoch, targets,max_size=5)
        meta_clid_results_accu,best_checkpoints_accu = update_best_results_dict(best_checkpoints_accu,meta_loss, test_outputs,epoch, targets,max_size=5)
        
        if test_acc >= best_acc:
            best_acc = test_acc
        sch_lr_2.step()
        if args.meta_goal=='ce_sloss' or args.meta_goal=='mae_sloss':
            noise_ratio, meta_set = eval_train(model_2, train_loader_2)
            train_meta_loader_2 = torch.utils.data.DataLoader(
                    meta_set, batch_size=args.meta_bsz, shuffle=True,
                    num_workers=args.prefetch, pin_memory=True)
            print(f'Meta_set_noise:{noise_ratio}')
            print(f'Meta_set_noise:{noise_ratio}',file=mytxt)  
            
        print('epoch_end:',meta_clid_results,)
        print('epoch_accu:',meta_clid_results_accu)
        print(
            "epoch:[%d/%d]\t train_loss:%.4f\t meta_loss_accu:%.4f\t train_acc:%.4f\t meta_acc_accu:%.4f\t test_acc:%.4f\t meta_loss:%.4f\t meta_acc:%.4f\t sve:%.4f\t lid:%.4f\t lsvr:%.4f\t cov:%.4f\t" % (
                (epoch + 1), args.epochs, train_loss, meta_loss, acc_train, acc_meta, test_acc,meta_clid,meta_acc,sve, lid,lsvr,cov))
        print('epoch_end:',meta_clid_results,file=mytxt)
        print('epoch_accu:',meta_clid_results_accu,file=mytxt)
        print(
            "epoch:[%d/%d]\t train_loss:%.4f\t meta_loss_accu:%.4f\t train_acc:%.4f\t meta_acc_accu:%.4f\t test_acc:%.4f\t meta_loss:%.4f\t meta_acc:%.4f\t sve:%.4f\t lid:%.4f\t lsvr:%.4f\t cov:%.4f\t" % (
                (epoch + 1), args.epochs, train_loss, meta_loss, acc_train, acc_meta, test_acc,meta_clid,meta_acc,sve, lid,lsvr,cov),file=mytxt)


    print('best_acc: ', best_acc)
    print('best_acc: ', best_acc, file=mytxt)


if __name__ == '__main__':
    save_path_dir = './exp_results_new_/'
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)
    # Configure logging
    txt_name = f'wnet_{args.scheduler}_{args.meta_goal}-{args.dataset}_{args.corruption_type}_\
               {args.corruption_prob}_lr_{args.lr}_meta_lr_{args.meta_lr}_es_{args.meta_bsz}_tau{args.tau}_seed{args.seed}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'

    # txt_name = f'ensemble_single_model-{args.dataset}_{args.corruption_type}_\
    #            {args.corruption_prob}_lr_{args.lr}_meta_lr_{args.meta_lr}_tau{args.tau}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    # txt_name = 'single_model-' + args.dataset + '_' + args.corruption_type + '_' \
    #            + str(args.corruption_prob) + '_' +'lr'+'_'+str(args.lr)+'_'+'meta_lr'+str(args.meta_lr)+'_'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # logging.basicConfig(
    #     filename=f'exp_results/{txt_name}'
    #     level=logging.INFO, 
    #     format='%(asctime)s - %(levelname)s - %(message)s')
    # print(txt_name)
    mytxt = open(save_path_dir + txt_name + '.txt', mode='w', encoding='utf-8')
    print(args) 

    print(args, file=mytxt) 

    main(args,mytxt)
    mytxt.close()
