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
import dataloader_red_mini_imagenet as mini_imagenet_dataloader
from sklearn.mixture import GaussianMixture
from PreResNet import *
from utils import *
import datetime
import warnings

# Disable all warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--id', default='redmini_vri')
parser.add_argument('--data_path', default='/home/rhu/r_work/VRI/VRI_DivideMix/data/red_mini_imagenet_s32', type=str, help='path to dataset')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=100, type=int)
parser.add_argument('--num_batches', default=1000, type=int)

parser.add_argument('--log_wandb', default=False, type=bool)
parser.add_argument('--wandb_project', default='', type=str)
parser.add_argument('--wandb_experiment', default='', type=str)
parser.add_argument('--wandb_entity', default='',type=str)
parser.add_argument('--wandb_resume', default=False, type=bool)
parser.add_argument('--need_clean', default=False, type=bool)
parser.add_argument('--single_meta', default=0, type=int)
parser.add_argument('--moco_pretrained', default=None, type=str)
parser.add_argument('--warmup', default=30, type=int)
parser.add_argument('--cos_lr', default=False, type=bool)
parser.add_argument('--lam', type=float, default=0.01)
parser.add_argument('--meta_goal', type=str, default='clid',help='ce,ce_sloss, mae,mae_sloss,clid')
parser.add_argument('--tau', default=0.5, type=float, help='tau for clid loss')
parser.add_argument('--meta_lr', '--meta_lr', default=0.02, type=float, help='earning rate for meta model')
parser.add_argument('--meta_bsz', default=100, type=int, help='meta batchsize')
parser.add_argument('--split', type=str, default='red_noise_nl_0.4')

args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
print(args)
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
            inputs_u, inputs_u2,labels_u,w_u = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2,labels_u,w_u = next(unlabeled_train_iter)
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)       
        labels_u = torch.zeros(inputs_u.size(0), args.num_class).scatter_(1, labels_u.view(-1,1), 1)        

        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2,labels_u,w_u = inputs_u.cuda(), inputs_u2.cuda(),labels_u.cuda(), w_u.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)            
###
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1)) / 2
            pu = w_u.float().view(-1, 1)*labels_u + (1-w_u).float().view(-1, 1)*pu
            ptu = pu**(1/args.T) # temparature sharpening
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()
            guess_u = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4
            guess_u = guess_u**(1/args.T)
            guess_t_u = guess_u/guess_u.sum(dim=1, keepdim=True)
            guess_t_u = guess_t_u.detach()
####                        
            
            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()  

            outputs_x21 = net2(inputs_x)
            outputs_x22 = net2(inputs_x2) 
            guess_x = (torch.softmax(outputs_x , dim=1) + torch.softmax(outputs_x2, dim=1) + torch.softmax(outputs_x21, dim=1) + torch.softmax(outputs_x2, dim=1)) / 4
            guess_x = guess_x**(1/args.T)
            guess_t_x = guess_x/guess_x.sum(dim=1, keepdim=True)
            guess_t_x = guess_t_x.detach()        
        
        
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)        
        
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)
        guess_target = torch.cat([guess_t_x, guess_t_x, guess_t_u, guess_t_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b
        
        guess_target_a, guess_target_b = guess_target, guess_target[idx]
        mixed_guess_target = l * guess_target_a + (1-l) * guess_target_b
    
        t=epoch+batch_idx/num_iter
        vnet = meta_step(net, vnet,optimizer, optimizer_vnet,mixed_input, mixed_target, mixed_guess_target,meta_loader,args,batch_size,t,args.warmup)

        outputs,feat = net(mixed_input,feat=True)
        
        logits_x = outputs[:batch_size*2]
        logits_u = outputs[batch_size*2:]
        # feat_x = feat[:batch_size*2]
        # feat_u = feat[batch_size*2:]
        with torch.no_grad():
            dict0 = {'feature': feat.detach(), 'target': mixed_target}
            _, _, w_new = vnet(dict0)
        
        Lx, Lu, lamb = criterion(w_new[:batch_size*2] *logits_x, mixed_target[:batch_size*2], w_new[batch_size*2:]*logits_u, mixed_guess_target[batch_size*2:],t, args.warmup)
        
        # Lx = -torch.mean(torch.sum(F.log_softmax(outputs*w_new, dim=1) * mixed_target, dim=1))
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(outputs, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))
       
        loss = Lx + lamb * Lu + penalty
        
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
            'unlabeled_loss':Lu.item()
        }
        if has_wandb and args.log_wandb:
            wandb.log(log_metric)
    
def warmup(net,optimizer,dataloader):
    net.train()
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)              
        loss = CEloss(outputs, labels)  
        
        penalty = conf_penalty(outputs)
        L = loss + penalty       
        L.backward()  
        optimizer.step() 

        # sys.stdout.write('\r')
        # sys.stdout.write('|Warm-up: Iter[%3d/%3d]\t CE-loss: %.4f  Conf-Penalty: %.4f'
        #         %(batch_idx+1, args.num_batches, loss.item(), penalty.item()))
        # sys.stdout.flush()
def warmup_vri(net,vnet, optimizer,dataloader,optimizer_vnet,meta_loader,args):
    net.train()
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()

        t=0
        batch_size=0
        vnet = meta_step(net, vnet,optimizer, optimizer_vnet,inputs, labels, meta_loader,args,batch_size,t,args.warmup,one=True)

        outputs,feat = net(inputs,feat=True)
        

        with torch.no_grad():
            dict0 = {'feature': feat.detach(), 'target': labels}
            _, _, w_new = vnet(dict0,one=True)
        
        # Lx, Lu, lamb = criterion(w_newš:batch_size*2] *logits_x, mixed_target[:batch_size*2], w_new[batch_size*2:]*logits_u, mixed_target[batch_size*2:],t, args.warmup)
        Lx = CEloss(w_new*outputs, labels)  

        # Lx = -torch.mean(torch.sum(F.log_softmax(outputs*w_new, dim=1) * labels, dim=1))
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(outputs, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))
       
        loss = Lx + penalty
        loss.backward()  
        optimizer.step()
        # sys.stdout.write('\r')
        # sys.stdout.write('|Warm-up: Iter[%3d/%3d]\t CE-loss: %.4f  Conf-Penalty: %.4f'
        #         %(batch_idx+1, args.num_batches, Lx.item(), penalty.item()))
        # sys.stdout.flush()

def meta_step(net, vnet, optimizer, optimizer_vnet,image, targets, metaloader,args,batch_size,t,warm_up,one=False):
        targets=targets.cuda()
    
        with higher.innerloop_ctx(net, optimizer) as (meta_net, meta_optimizer):

            for s in range(1):
                outputs, feat  = meta_net(image,feat=True)
                dict0 = {'feature':feat.detach(), 'target':targets}
                mean, log_var, v_lambda = vnet(dict0,one)
                dict1 = {'feature':feat.detach()}
                mean_p, log_var_p, _ = vnet(dict1,one)
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

                l_f_meta = Lx + args.lam * kl_loss(mean, log_var, mean_p, log_var_p) +penalty
                
                meta_optimizer.step(l_f_meta)
            try:
                batch = next(loader_iter)
            except:
                # If the iterator is exhausted, reinitialize it
                loader_iter = iter(metaloader)
                batch = next(loader_iter)
                
            if args.meta_goal=='clid':
                val_data, val_labels = batch
            else:
                val_data, _, val_labels, _ = batch
                
            val_data = val_data.cuda()
            val_labels = val_labels.cuda()
            y_g_hat,feat_val = meta_net(val_data,feat=True)

            if args.meta_goal=='ce':
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


def val(net,val_loader,k):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
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

def test(net1,net2,test_loader):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)       
            outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                    
    acc = 100.*correct/total
    print("\n| Test Acc: %.2f%%\n" %(acc))
    if has_wandb:
        if args.log_wandb:
            wandb.log({'test_acc': acc})
    return acc    
    
def eval_train(model,eval_loader,all_loss):
    model.eval()
    
    losses = torch.zeros(50010)
    n=0
    with torch.no_grad():
        for batch_idx, (inputs, targets, path) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[n]=loss[b] 
                # paths.append(path[b])
                n+=1
        
    losses = (losses-losses.min())/(losses.max()-losses.min())   
    all_loss.append(losses)

    losses = losses.reshape(-1,1)
    gmm = GaussianMixture(n_components=2,max_iter=10,reg_covar=5e-4,tol=1e-2)
    gmm.fit(losses)
    prob = gmm.predict_proba(losses) 
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
    


def split_pseudo_clean_set(model, train_loader, eval_loader, all_loss, p_threshold, meta_threshold):
    prob, all_loss = eval_train(model, eval_loader, all_loss)
    
    # generate meta set
    meta_pred = (prob > meta_threshold)
    train_meta_loader  = train_loader.run('labeled', meta_pred, prob)
    
    # generate train set
    pred = (prob > p_threshold)
    
    # ## prevent from bad cases
    # if pred.sum() < 64 or pred.sum() > 45000:
    #      pred = (prob > np.median(prob))
            
    labeled_trainloader, unlabeled_trainloader = train_loader.run('train', pred, prob)
    
    return labeled_trainloader, unlabeled_trainloader, train_meta_loader, all_loss

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))
               
def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model  


current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
log=open(f'/home/rhu/r_work/VRI/VRI_DivideMix/checkpoint/metaselected_{args.meta_goal}_{args.id}_{args.seed}_{args.meta_lr}_{formatted_time}.txt','w')
log.flush()
args.train_split_file = 'noise_0.4'
args.train_path = '{}/training_s32/{}'.format(args.data_path, args.split)
args.test_split_file = 'clean_validation'
args.test_path = '{}/validation_s32/'.format(args.data_path)


loader = mini_imagenet_dataloader.red_mini_imagenet_dataloader(args.train_split_file, batch_size=args.batch_size, num_workers=6, root_dir=args.train_path)
test_loader = mini_imagenet_dataloader.red_mini_imagenet_dataloader(args.test_split_file, batch_size=args.batch_size, num_workers=2, root_dir=args.test_path)


meta_loader_1= loader.run('meta_clid')
meta_loader_2= loader.run('meta_clid')


print('| Building net')
net1 = create_model()
net2 = create_model()
# vnet1 = vri(576, 1024, 512, args.num_class).cuda()
# vnet2 = vri(576, 1024, 512, args.num_class).cuda()

vnet1 = vri(576, 1024, 512, args.num_class).cuda()
vnet2 = vri(576, 1024, 512, args.num_class).cuda()

cudnn.benchmark = True

optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4,nesterov=True)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4,nesterov=True)

optimizer_vnet1 = torch.optim.Adam(vnet1.parameters(), args.meta_lr, weight_decay=5e-4)
optimizer_vnet2 = torch.optim.Adam(vnet2.parameters(), args.meta_lr, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()
criterion = SemiLoss()




# from torch.utils.data import Dataset, DataLoader
# meta_loader_1 = DataLoader(
#                 dataset=meta_loader_1,
#                 batch_size=args.batch_size,
#                 shuffle=True,
#                 num_workers=5)

# meta_loader_2 = DataLoader(
#                 dataset=meta_loader_2,
#                 batch_size=args.batch_size,
#                 shuffle=True,
#                 num_workers=5)

best_acc = [0,0]
scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=args.num_epochs)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=args.num_epochs)
all_loss = [[],[]] # save the history of losses from two networks
eval_loader = loader.run('eval_train')  # evaluate training data loss for next epoch  
best_checkpoints1={}
t_loader = test_loader.run('test')

for epoch in range(args.num_epochs+args.warmup):   
    lr=args.lr
    if args.cos_lr:
        scheduler1.step()
        scheduler2.step()
    else:

        lr = args.lr * ((0.1 ** int(epoch >= 200)) * (0.1 ** int(epoch >= 250)))
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr            
        
        train_loader = loader.run('warmup')

        
    if epoch<args.warmup:     # warm up
        train_loader = loader.run('warmup')
        warmup(net1,optimizer1,train_loader)     
               
    else:
        warmup_vri(net1,vnet1, optimizer1,train_loader,optimizer_vnet1,meta_loader_1,args)
        meta_acc_1,meta_clid_1= test_meta(model=net1, meta_loader=meta_loader_1,meta_goal=args.meta_goal,args=args)
    
        test_acc_1, test_outputs_1,targets_1 = test_m(model=net1, test_loader=t_loader)
    
        meta_clid_results1,best_checkpoints1 = update_best_results_dict(best_checkpoints1,meta_clid_1, test_outputs_1,epoch, targets_1,max_size=5)
    
    
        print('Ensemble ACC1:',meta_clid_results1)         
    #     # pred1 = (prob1 > args.p_threshold)  # divide dataset  
    #     # pred2 = (prob2 > args.p_threshold)      
        
    #     print('\n\nTrain Net1')
    #     # labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2) # co-divide
    #     labeled_trainloader2, unlabeled_trainloader2, meta_loader2, all_loss[0] = split_pseudo_clean_set(net1, loader, eval_loader, all_loss[0], 0.5, 0.5)
    #     labeled_trainloader1, unlabeled_trainloader1, meta_loader1, all_loss[1] = split_pseudo_clean_set(net2, loader, eval_loader, all_loss[1], 0.5, 0.5)

    #     if args.meta_goal=='clid':
    #         first_meta = meta_loader_1
    #         second_meta = meta_loader_2
    #     else:
    #         first_meta = meta_loader1
    #         second_meta = meta_loader2
 
        
    #     train(epoch,net1,net2,optimizer1,optimizer_vnet1,labeled_trainloader1, unlabeled_trainloader1,vnet1,first_meta,args)              # train net1
    #     print('\nTrain Net2')


    #     train(epoch,net2,net1,optimizer2,optimizer_vnet2,labeled_trainloader2, unlabeled_trainloader2,vnet2,second_meta,args)              # train net2
    # val_loader = test_loader.run('test') # validation
    # acc1 = val(net1,val_loader,1)
    # acc2 = val(net2,val_loader,2)   
    # log.write('Validation Epoch:%d      Acc1:%.2f  Acc2:%.2f\n'%(epoch,acc1,acc2))
    # log.flush() 
    # print('\n==== net 1 evaluate next epoch training data loss ====') 
    # eval_loader = loader.run('eval_train')  # evaluate training data loss for next epoch  
    # prob1,paths1 = eval_train(epoch,net1) 
    # # print('\n==== net 2 evaluate next epoch training data loss ====') 
    # eval_loader = loader.run('eval_train')  
    # prob2,paths2 = eval_train(epoch,net2) 

    acc1 = val(net1,t_loader,1)

# net1.load_state_dict(torch.load('/home/ec2-user/disk/checkpoint/%s_net1.pth.tar'%args.id))
# net2.load_state_dict(torch.load('/home/ec2-user/disk/checkpoint/%s_net2.pth.tar'%args.id))
    print(f"\n==== Test ACC:{acc1} ====") 

