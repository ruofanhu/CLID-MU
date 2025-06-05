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
import dataloader_clothing1M as dataloader
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
from dataset_food101 import Food101N,split_meta_train
import torchvision.transforms as transforms
import random

parser = argparse.ArgumentParser(description='PyTorch Food101N Training')
parser.add_argument('--batch_size', default=128, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.002, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=160, type=int)
parser.add_argument('--id', default='food101n_vri')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=101, type=int)
parser.add_argument('--num_batches', default=3000, type=int)
parser.add_argument('--data_path', default='/home/rhu/r_work/VRI/data/Food101N', type=str, help='path to dataset')
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
parser.add_argument('--model', type=str, default='resnet50',help='vgg or resnet50')

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

        try:
            meta_batch = next(meta_loader_iter)
        except:
            # If the iterator is exhausted, reinitialize it
            loader_iter = iter(meta_loader)
            meta_loader_iter = iter(meta_loader)
            meta_batch = next(meta_loader_iter) 
        t=0    
        batch_size=0

        vnet = meta_step(net, vnet,optimizer, optimizer_vnet,inputs, labels, meta_batch,args,batch_size,t,args.warmup,one=True)

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
        for batch_idx, (inputs, targets,_) in enumerate(val_loader):
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


    
def eval_train(model, train_loader):
    num_meta=1000
    model.eval()
    data_ids=[]
    all_loss =[]
    y_train = []
    train_set = train_loader.dataset
    with torch.no_grad():
        for batch_idx, (inputs,targets, ids ) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, feat = model(inputs)
            loss = F.cross_entropy(outputs, targets,reduction='none')
            y_train.extend(targets.cpu().tolist())
            all_loss.extend(loss.cpu().numpy())
            data_ids.extend(ids.cpu().numpy().tolist())
    
    num_classes=outputs.shape[1]        
    data_info = pd.DataFrame({'dataset_ids':data_ids,'label': y_train,'loss': all_loss})
    # Step 3: Select top samples with smallest losses for each class
    selected_indices = []
    samples_per_class = num_meta // num_classes  # Assuming 10 classes

    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    input_loss= data_info['loss'].values.reshape(-1, 1)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]
    pred = (prob > 0.5)
    data_info['clean_prob']=prob

    for class_label in range(num_classes):  # Assuming class labels are 0 to 9
        class_samples = data_info[data_info['label'] == class_label]  # Filter by class
        # class_samples_sorted = class_samples.sort_values(by=['loss','max_probability'],ascending=[True,False])  # Sort by loss
        class_samples_sorted = class_samples.sort_values(by='clean_prob',ascending=False)  # Sort by loss
        selected_indices.extend(class_samples_sorted.head(samples_per_class)['dataset_ids'].tolist())  # Collect indices
    
    # Step 4: Create a subset of the original dataset
    selected_data_info = data_info[data_info['dataset_ids'].isin(selected_indices)]
    random.shuffle(selected_indices)
    meta_dset = CustomSubset(train_set,selected_indices)
    reset_subset = CustomSubset(meta_dset, range(len(meta_dset)))
    meta_loader = DataLoader(
        dataset=reset_subset, 
        batch_size=args.meta_bsz,
        shuffle=True,
        num_workers=4, pin_memory=True)
    return meta_loader
    
def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)



def meta_step(net, vnet, optimizer, optimizer_vnet,image, targets, meta_batch,args,batch_size,t,warm_up,one=True):
        targets=targets.cuda()
    
        with higher.innerloop_ctx(net, optimizer) as (meta_net, meta_optimizer):

            for s in range(1):
                # outputs, feat  = meta_net(image,feat=True)                
                outputs, feat = meta_net(image)

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

                l_f_meta = Lx + args.lam * kl_loss(mean, log_var, mean_p, log_var_p)
                
                meta_optimizer.step(l_f_meta)

            val_data, val_labels,ids = meta_batch

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
    
def unfreeze_encoder(model):
    for name, param in model.named_parameters():
        if not name.startswith('fc'):
            param.requires_grad = True
    print("Encoder unfrozen.")


current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
log=open(f'/home/rhu/r_work/VRI/VRI_DivideMix/checkpoint/food11n_{args.model}_{args.meta_goal}_{args.meta_lr}_{args.tau}_{formatted_time}_{args.warmup}.txt','w')
log.flush()


# -- specify dataset
# data augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
num_workers=4

dataset_pth=f"{args.data_path}/loaded_dataset.pt"
if os.path.exists(dataset_pth):
    data_list=torch.load(dataset_pth)
    trainset=data_list[0]
    testset = data_list[1]
else:
    trainset = Food101N(data_path=args.data_path, split='train', transform=transform_train)
    testset = Food101N(data_path=args.data_path, split='test', transform=transform_test)
    torch.save([trainset,testset],dataset_pth)

meta_set=split_meta_train(trainset,2000,args.meta_goal,num_classes=101)
meta_loader_1 = torch.utils.data.DataLoader(meta_set, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers,drop_last=True)

val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size * 4, shuffle=False, num_workers=num_workers)

num_class = 101



print('| Building net')
net1 = create_model()

vnet1 = vri(2112, 1024, 512, args.num_class).cuda()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr)

cudnn.benchmark = True



optimizer_vnet1 = torch.optim.Adam(vnet1.parameters(), args.meta_lr, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()
criterion = SemiLoss()



best_acc = [0,0]
scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=args.num_epochs,eta_min=0.0001, last_epoch=-1) #args.num_epochs

best_checkpoints1={}



print(args)
unfreeze_encoder(net1)

for epoch in range(args.num_epochs+args.warmup):   
    
    if epoch==args.warmup:
        # unfreeze_encoder(net1)
        optimizer1 = optim.Adam(net1.parameters(), lr=args.lr)
    
    if epoch<args.warmup:
        
        warmup(net1,optimizer1,train_loader)
        acc1 = val(net1,val_loader,1)
        print(f"\n==== Epoch {epoch}, Test ACC:{acc1} ====\n") 
        
    else:
        lr=args.lr

        # if args.meta_goal!='ce_sloss':

        # else:
        # meta_loader_1 = eval_train(net1, train_loader)

        # print('Warmup train Net1')
        train_vri(net1,vnet1, optimizer1,train_loader,optimizer_vnet1,meta_loader_1,args)
        # train_loader = loader.run('warmup')
        # print('\nWarmup train Net2')
        # warmup_vri(net2,vnet1, optimizer2,train_loader,optimizer_vnet2,meta_loader_2,args)
        acc1 = val(net1,val_loader,1)
        print(f"\n==== Epoch:{epoch} Test ACC:{acc1} ====\n") 
        # log.write('Validation Epoch:%d    Acc1:%.2f\n'%(epoch,acc1))
    
        meta_acc_1,meta_clid_1= test_meta(model=net1, meta_loader=meta_loader_1,meta_goal=args.meta_goal,args=args)
        test_acc_1, test_outputs_1,targets_1 = test_m_food(model=net1, test_loader=val_loader)
        meta_clid_results1,best_checkpoints1 = update_best_results_dict(best_checkpoints1,meta_clid_1, test_outputs_1,epoch, targets_1,max_size=5)
        print(f"Ensenble {meta_clid_results1}")
        
        log.write(f'Test Epoch:{epoch}  Acc1:{acc1}\n')
        log.write(f'{json.dumps(meta_clid_results1, indent=4)}\n\n')    
        if args.cos_lr:
            scheduler1.step()
        else:
            # if epoch >= 50:
            #     lr /= 10
            lr = args.lr * ((0.1 ** int(epoch >= 100)) * (0.1 ** int(epoch >= 150)))       
     
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr
 
        # if epoch % 5 == 0:
        #     train_loss = mixup(net1, train_meta_loader_2, optimizer1, args.alpha, args.num_class)
        #     test_acc = test(model=model_2, test_loader=test_loader)
        
