from __future__ import print_function
import sys

import higher
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models as models
import random
import os
import argparse
import numpy as np
import dataloader_animal as dataloader
from sklearn.mixture import GaussianMixture
from PreResNet import *
from resnet50 import resnet50_pre
from utils import *
import datetime
from models.resnet import resnet50
from rand_aug import RandAugmentMC,RandAugmentwogeo
from vgg import vgg19bn
from dataloader_animal import CustomSubset
from torch.utils.data import DataLoader
import json
import pandas as pd
import time
import datetime
parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')
parser.add_argument('--batch_size', default=128, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.002, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=150, type=int)
parser.add_argument('--id', default='clothing1m_vri')
parser.add_argument('--data_path', default='../data', type=str, help='path to dataset')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--num_batches', default=3000, type=int)

parser.add_argument('--log_wandb', default=False, type=bool)
parser.add_argument('--wandb_project', default='', type=str)
parser.add_argument('--wandb_experiment', default='', type=str)
parser.add_argument('--wandb_entity', default='',type=str)
parser.add_argument('--wandb_resume', default=False, type=bool)
parser.add_argument('--need_clean', default=False, type=bool)
parser.add_argument('--single_meta', default=0, type=int)
parser.add_argument('--moco_pretrained', default='/home/rhu/r_work/VRI/VRI_DivideMix/pretrained/ckpt_clothing_resnet50.pth', type=str)
parser.add_argument('--warmup', default=5, type=int)
parser.add_argument('--cos_lr', default=False, type=bool)
parser.add_argument('--lam', type=float, default=0.01)
parser.add_argument('--meta_goal', type=str, default='clid',help='ce,ce_sloss, mae,mae_sloss,clid')
parser.add_argument('--tau', default=0.5, type=float, help='tau for clid loss')
parser.add_argument('--meta_lr', '--meta_lr', default=0.02, type=float, help='earning rate for meta model')
parser.add_argument('--meta_bsz', default=100, type=int, help='meta batchsize')
parser.add_argument('--model', type=str, default='vgg',help='vgg or resnet50')

args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

try:
    import wandb
    has_wandb = True
   # os.environ["WANDB_API_KEY"] = YOUR_KEY_HERE
except ImportError:
    has_wandb = False
    print('please install wandb')
if args.log_wandb:
    if has_wandb:
        wandb.init(project=str(args.wandb_project), name=str(args.wandb_experiment),
                       entity=str(args.wandb_entity), resume=args.wandb_resume)
        wandb.config.update(args)




# Training
def train(epoch,net,net2,optimizer,optimizer_vnet,labeled_trainloader,unlabeled_trainloader, vnet,meta_loader,args):
    net.train()
    net2.eval() #fix one network and train the other
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11,_ = net(inputs_u)
            outputs_u12,_ = net(inputs_u2)
            outputs_u21,_ = net2(inputs_u)
            outputs_u22,_ = net2(inputs_u2)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x,_ = net(inputs_x)
            outputs_x2,_ = net(inputs_x2)            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)        
        
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a[:batch_size*2] + (1 - l) * input_b[:batch_size*2]        
        mixed_target = l * target_a[:batch_size*2] + (1 - l) * target_b[:batch_size*2]

        t=epoch+batch_idx/num_iter
        vnet = meta_step(net, vnet,optimizer, optimizer_vnet,mixed_input, mixed_target, meta_loader,args,batch_size,t,args.warmup)

        outputs,feat = net(mixed_input)
        
        logits_x = outputs[:batch_size*2]
        logits_u = outputs[batch_size*2:]
        # feat_x = feat[:batch_size*2]
        
        # feat_u = feat[batch_size*2:]
        with torch.no_grad():
            dict0 = {'feature': feat.detach(), 'target': mixed_target}
            _, _, w_new = vnet(dict0)
        
        # Lx, Lu, lamb = criterion(w_new[:batch_size*2] *logits_x, mixed_target[:batch_size*2], w_new[batch_size*2:]*logits_u, mixed_target[batch_size*2:],t, args.warmup)
        
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs*w_new, dim=1) * mixed_target, dim=1))
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(outputs, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))
       
        loss = Lx + penalty
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if meta_loader is not None:
        #     net = l2b(net, optimizer, input_a,  target_a, metaloader=meta_loader)

        # sys.stdout.write('\r')
        # sys.stdout.write('Clothing1M | Epoch [%3d/%3d] Iter[%3d/%3d]\t  Labeled loss: %.4f '
        #         %(epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item()))
        # sys.stdout.flush()
        log_metric = {
            'laebled_loss': Lx.item(),
        }
        if has_wandb and args.log_wandb:
            wandb.log(log_metric)
    
def warmup(net,optimizer,dataloader):
    net.train()
    for batch_idx, (inputs, labels, idxs) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs,_ = net(inputs)              
        loss = CEloss(outputs, labels)  
        
        loss.backward()  
        optimizer.step() 

        # sys.stdout.write('\r')
        # sys.stdout.write('|Warm-up: Iter[%3d/%3d]\t CE-loss: %.4f  Conf-Penalty: %.4f'
        #         %(batch_idx+1, args.num_batches, loss.item(), penalty.item()))
        # sys.stdout.flush()

def train_vri(net,vnet, optimizer,dataloader,optimizer_vnet,meta_loader,args):
    net.train()
    for batch_idx, (inputs, labels, idxs) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 


        t=0
        batch_size=0
        vnet = meta_step(net, vnet,optimizer, optimizer_vnet,inputs, labels, meta_loader,args,batch_size,t,args.warmup,one=True)

        outputs,feat = net(inputs)
        

        with torch.no_grad():
            dict0 = {'feature': feat.detach(), 'target': labels}
            _, _, w_new = vnet(dict0,one=True)
        
        # Lx, Lu, lamb = criterion(w_new[:batch_size*2] *logits_x, mixed_target[:batch_size*2], w_new[batch_size*2:]*logits_u, mixed_target[batch_size*2:],t, args.warmup)
        Lx = CEloss(w_new*outputs, labels)  

        # Lx = -torch.mean(torch.sum(F.log_softmax(outputs*w_new, dim=1) * labels, dim=1))
        
        # regularization
        # prior = torch.ones(args.num_class)/args.num_class
        # prior = prior.cuda()        
        # pred_mean = torch.softmax(outputs, dim=1).mean(0)
        # penalty = torch.sum(prior*torch.log(prior/pred_mean))
       
        # loss = Lx + penalty
        loss = Lx

        optimizer.zero_grad()
        loss.backward()  
        optimizer.step()

def val(net,val_loader,k):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs,_ = net(inputs)
            _, predicted = torch.max(outputs, 1)         
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()              
    acc = 100.*correct/total
    print("\n| Test\t Net%d  Acc: %.2f%%" %(k,acc))
    # if acc > best_acc[k-1]:
    #     best_acc[k-1] = acc
    #     print('| Saving Best Net%d ...'%k)
    #     save_point = '/home/rhu/r_work/VRI/VRI_DivideMix/checkpoint/%s_net%d.pth.tar'%(args.id,k)
    #     torch.save(net.state_dict(), save_point)
    if has_wandb:
        if args.log_wandb:
            wandb.log({'eval_acc':acc})
    return acc

def test(epoch,net1,net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1,_ = net1(inputs)
            outputs2,_ = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    if has_wandb:
        if args.log_wandb:
            wandb.log({'test_acc':acc})
    log.flush() 
    
# def eval_train(model, train_loader):
#     num_meta=1000
#     model.eval()
#     data_ids=[]
#     all_loss =[]
#     y_train = []
#     train_set = train_loader.dataset
#     with torch.no_grad():
#         for batch_idx, (inputs,targets, ids ) in enumerate(train_loader):
#             inputs, targets = inputs.cuda(), targets.cuda()
#             outputs, feat = model(inputs)
#             loss = F.cross_entropy(outputs, targets,reduction='none')
#             y_train.extend(targets.cpu().tolist())
#             all_loss.extend(loss.cpu().numpy())
#             data_ids.extend(ids.cpu().numpy().tolist())
    
#     num_classes=outputs.shape[1]        
#     data_info = pd.DataFrame({'dataset_ids':data_ids,'label': y_train,'loss': all_loss})
#     # Step 3: Select top samples with smallest losses for each class
#     selected_indices = []
#     samples_per_class = num_meta // num_classes  # Assuming 10 classes

#     gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
#     input_loss= data_info['loss'].values.reshape(-1, 1)
#     gmm.fit(input_loss)
#     prob = gmm.predict_proba(input_loss) 
#     prob = prob[:,gmm.means_.argmin()]
#     pred = (prob > 0.5)
#     data_info['clean_prob']=prob

#     for class_label in range(num_classes):  # Assuming class labels are 0 to 9
#         class_samples = data_info[data_info['label'] == class_label]  # Filter by class
#         # class_samples_sorted = class_samples.sort_values(by=['loss','max_probability'],ascending=[True,False])  # Sort by loss
#         class_samples_sorted = class_samples.sort_values(by='clean_prob',ascending=False)  # Sort by loss
#         selected_indices.extend(class_samples_sorted.head(samples_per_class)['dataset_ids'].tolist())  # Collect indices
    
#     # Step 4: Create a subset of the original dataset
#     selected_data_info = data_info[data_info['dataset_ids'].isin(selected_indices)]
#     random.shuffle(selected_indices)
#     meta_dset = CustomSubset(train_set,selected_indices)
#     reset_subset = CustomSubset(meta_dset, range(len(meta_dset)))
#     meta_loader = DataLoader(
#         dataset=reset_subset, 
#         batch_size=args.meta_bsz,
#         shuffle=True,
#         num_workers=4, pin_memory=True)
#     return meta_loader

def eval_train(model,all_loss):    
    model.eval()
    losses = torch.zeros(50000)    
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs,_ = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]         
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)

    # if args.r==0.9: # average loss over last 5 epochs to improve convergence stability
    #     history = torch.stack(all_loss)
    #     input_loss = history[-5:].mean(0)
    #     input_loss = input_loss.reshape(-1,1)
    # else:
    input_loss = losses.reshape(-1,1)
    
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]         
    return prob,all_loss    
def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

def meta_step(net, vnet, optimizer, optimizer_vnet,image, targets, metaloader,args,batch_size,t,warm_up,one=False):
        targets=targets.cuda()
    
        with higher.innerloop_ctx(net, optimizer) as (meta_net, meta_optimizer):

            for s in range(1):
                # outputs, feat  = meta_net(image,feat=True)                
                outputs, feat = meta_net(image)

                dict0 = {'feature':feat.detach(), 'target':targets}
                mean, log_var, v_lambda = vnet(dict0)
                dict1 = {'feature':feat.detach()}
                mean_p, log_var_p, _ = vnet(dict1)
                if one:
                    targets_onehot = torch.nn.functional.one_hot(targets, args.num_class).float().cuda()

                    Lx = -torch.mean(torch.sum(F.log_softmax(v_lambda*outputs, dim=1) * targets_onehot, dim=1))

                else:
                    Lx = -torch.mean(torch.sum(F.log_softmax(v_lambda*outputs, dim=1) * targets, dim=1))

                #regularization
                prior = torch.ones(args.num_class)/args.num_class
                prior = prior.cuda()        
                pred_mean = torch.softmax(outputs, dim=1).mean(0)
                penalty = torch.sum(prior*torch.log(prior/pred_mean))

                l_f_meta = Lx + args.lam * kl_loss(mean, log_var, mean_p, log_var_p)
                
                meta_optimizer.step(l_f_meta)
            try:
                batch = next(loader_iter)
            except:
                # If the iterator is exhausted, reinitialize it
                loader_iter = iter(metaloader)
                batch = next(loader_iter)
            if args.meta_goal=='ce_sloss':
                val_data, val_labels,ids = batch
            else:
                val_data, val_labels = batch
                
            val_data = val_data.cuda()
            val_labels = val_labels.cuda()
            y_g_hat,feat_val = meta_net(val_data)

            if args.meta_goal=='ce' or args.meta_goal=='ce_sloss':
                l_g_meta = F.cross_entropy(y_g_hat,val_labels.long())
            elif args.meta_goal=='mae' or args.meta_goal=='mae_sloss':
                l_g_meta = mae_loss(y_g_hat, val_labels.long())
            ### Apply clid_loss
            elif args.meta_goal=='clid':
                l_g_meta = clid_loss(feat_val,y_g_hat,args.tau)
            optimizer_vnet.zero_grad()
            l_g_meta.backward()
            optimizer_vnet.step()

        return vnet



class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))
               
# def create_model(num_classes=14):
#     # chekpoint = torch.load(args.moco_pretrained)
#     chekpoint = torch.load('/home/rhu/r_work/VRI/VRI_DivideMix/pretrained/ckpt_clothing_resnet50.pth')
#     sd = {}
#     for ke in chekpoint['model']:
#         nk = ke.replace('module.', '')
#         sd[nk] = chekpoint['model'][ke]
#     model = SupCEResNet(net='resnet50', num_class=(num_classes, pool=True)
#     model.load_state_dict(sd, strict=False)
#     model = model.cuda()
#     return model


def create_model():
    if args.model!='vgg':
        model = resnet50_pre(pretrained=True,num_classes=args.num_class)
        model.fc = nn.Linear(2048,args.num_class)
    else:
        model=vgg19bn()
    model = model.cuda()
    return model     


current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
log=open(f'/home/rhu/r_work/VRI/VRI_DivideMix/checkpoint/animal_two_{args.model}_{args.meta_goal}_{args.meta_lr}_{args.tau}_{formatted_time}.txt','w')
log.flush()

loader = dataloader.animal_dataloader(root_dir=args.data_path,batch_size=args.batch_size,num_workers=5)
print('| Building net')
net1 = create_model()
net2 = create_model()
if args.model=='vgg':
    vnet1 = vri(576, 1024, 512, args.num_class).cuda()
    vnet2 = vri(576, 1024, 512, args.num_class).cuda()
    optimizer1 = optim.SGD(net1.parameters(), lr=args.lr,momentum=0.9,weight_decay=1e-3)
    optimizer2 = optim.SGD(net2.parameters(), lr=args.lr,momentum=0.9,weight_decay=1e-3)

else:
    vnet1 = vri(2112, 1024, 512, args.num_class).cuda()
    vnet2 = vri(2112, 1024, 512, args.num_class).cuda()
    optimizer1 = optim.Adam(net1.parameters(), lr=args.lr)
    optimizer2 = optim.Adam(net2.parameters(), lr=args.lr)

cudnn.benchmark = True



optimizer_vnet1 = torch.optim.Adam(vnet1.parameters(), args.meta_lr, weight_decay=5e-4)
optimizer_vnet2 = torch.optim.Adam(vnet2.parameters(), args.meta_lr, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()
criterion = SemiLoss()



best_acc = [0,0]
scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=args.num_epochs,eta_min=0.001, last_epoch=-1) #args.num_epochs
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=args.num_epochs,eta_min=0.001, last_epoch=-1)

best_checkpoints1={}
val_loader = loader.run('test') # validation
train_loader = loader.run('warmup')
print(args)


all_loss = [[],[]] # save the history of losses from two networks
best_checkpoints1={}
best_checkpoints2={}
meta_loader_1 = loader.run('meta')
meta_loader_2 = loader.run('meta')

for epoch in range(args.num_epochs):   
    start_time = time.time()
        
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')
    
    if epoch<args.warmup:       
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(net1,optimizer1,warmup_trainloader)    
        print('\nWarmup Net2')
        warmup(net2,optimizer2,warmup_trainloader) 
   
    else:         
        prob1,all_loss[0]=eval_train(net1,all_loss[0])   
        prob2,all_loss[1]=eval_train(net2,all_loss[1])          
               
        pred1 = (prob1 > args.p_threshold)      
        pred2 = (prob2 > args.p_threshold)      
        
        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2,  prob2) # co-divide

        if args.meta_goal=='clid':
            first_meta = meta_loader_1
            second_meta = meta_loader_2
        else:
            first_meta = labeled_trainloader
            
        train(epoch,net1,net2,optimizer1,optimizer_vnet1, labeled_trainloader, unlabeled_trainloader,vnet1,first_meta,args) # train net1
        scheduler1.step()    
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1) # co-divide
    
        if args.meta_goal!='clid':
            second_meta = labeled_trainloader 
        train(epoch,net2,net1,optimizer2,optimizer_vnet2,labeled_trainloader, unlabeled_trainloader,vnet2,second_meta,args) # train net2
        scheduler2.step()    

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        if args.meta_goal=='clid':
            meta_acc_1,meta_clid_1= test_meta(model=net1, meta_loader=first_meta,meta_goal=args.meta_goal,args=args)
            meta_acc_2,meta_clid_2= test_meta(model=net2, meta_loader=second_meta,meta_goal=args.meta_goal,args=args)
        else:
            meta_acc_1,meta_clid_1= test_meta_s(model=net1, meta_loader=first_meta,meta_goal=args.meta_goal,args=args)
            meta_acc_2,meta_clid_2= test_meta_s(model=net2, meta_loader=second_meta,meta_goal=args.meta_goal,args=args)
    
        test_acc_1, test_outputs_1,targets_1 = test_m(model=net1, test_loader=test_loader)
        test_acc_2, test_outputs_2,targets_2 = test_m(model=net2, test_loader=test_loader)
    
        
        meta_clid_results1,best_checkpoints1 = update_best_results_dict(best_checkpoints1,meta_clid_1, test_outputs_1,epoch, targets_1,max_size=5)
        meta_clid_results2,best_checkpoints2 = update_best_results_dict(best_checkpoints2,meta_clid_2, test_outputs_2,epoch, targets_2,max_size=5)
    
        acc_double = best_results_dict_double(best_checkpoints1,best_checkpoints2,targets_1,max_size=5)
    
        print('Ensemble ACC1:',meta_clid_results1,'\nEnsemble ACC2:',meta_clid_results2,'\n Double Ensemble:',acc_double)
    test(epoch,net1,net2)  
