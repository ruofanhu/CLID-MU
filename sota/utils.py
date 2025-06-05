import torch
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture
import random



def update_best_results_dict(best_checkpoints,meta_clid, outputs,epoch, targets,max_size=5):
    # Add the new checkpoint with its metric and epoch as the key
    best_checkpoints[epoch] = {'meta_clid': meta_clid, 'test_outputs': outputs}
    
    # Sort by metric (keep the top 5)
    sorted_checkpoints = sorted(best_checkpoints.items(), key=lambda x: x[1]['meta_clid'], reverse=False)  # Use reverse=False for loss
    best_checkpoints_5 = dict(sorted_checkpoints[:max_size])
    best_checkpoints_1 = dict(sorted_checkpoints[:1])
    # Combine the test outputs from all best checkpoints
    num_instances = len(next(iter(best_checkpoints_5.values()))['test_outputs'])  # Number of instances
    num_classes = len(next(iter(best_checkpoints.values()))['test_outputs'][0])  # Number of classes
    
    # Initialize a list to store the combined logits for each instance
    combined_logits = [np.zeros(num_classes) for _ in range(num_instances)]

    meta_clid_results={}
    # Add the logits across the best checkpoints
    for key,checkpoint in best_checkpoints_5.items():
        test_outputs = checkpoint['test_outputs']
        meta_clid_results[key]= checkpoint['meta_clid']
        for i in range(num_instances):
            combined_logits[i] += test_outputs[i]  # Element-wise addition
    
    # Determine the final class by selecting the max class for each instance
    final_predictions_5 = torch.tensor([np.argmax(logits) for logits in combined_logits])
    acc_5 = (final_predictions_5 == targets).sum().item()/targets.size(0)

############################################
    # Initialize a list to store the combined logits for each instance
    combined_logits = [np.zeros(num_classes) for _ in range(num_instances)]
    
    # Add the logits across the best checkpoints
    for checkpoint in best_checkpoints_1.values():
        test_outputs = checkpoint['test_outputs']
        for i in range(num_instances):
            combined_logits[i] += test_outputs[i]  # Element-wise addition
    
    # Determine the final class by selecting the max class for each instance
    final_predictions_1 = torch.tensor([np.argmax(logits) for logits in combined_logits])
    acc_1 = (final_predictions_1 == targets).sum().item()/targets.size(0)

    
    meta_clid_results['acc_5']=acc_5
    meta_clid_results['acc_1']=acc_1
              
    return meta_clid_results, best_checkpoints_5


def best_results_dict_double(best_checkpoints1,best_checkpoints2,targets,max_size=5):
    # Sort by metric (keep the top 5)
    merged_loss_values = []
    merged_predictions = []
    
    for d in [best_checkpoints1,best_checkpoints2]:
        for epoch_data in d.values():
            merged_loss_values.append(epoch_data["meta_clid"])
            merged_predictions.append(epoch_data["test_outputs"])
            
    # Combine loss values and predictions
    combined = list(zip(merged_loss_values, merged_predictions))
    
    # Sort by loss values in ascending order
    sorted_combined = sorted(combined, key=lambda x: x[0], reverse=False)
    
    # Extract sorted loss values and predictions
    sorted_loss_values = [x[0] for x in sorted_combined[:max_size]]
    sorted_predictions = [x[1] for x in sorted_combined[:max_size]]
    
    # Combine the test outputs from all best checkpoints
    num_instances = sorted_predictions[0].shape[0]  # Number of instances
    num_classes = sorted_predictions[0].shape[1]  # Number of classes
    
    # Initialize a list to store the combined logits for each instance
    combined_logits = [np.zeros(num_classes) for _ in range(num_instances)]
    
    meta_clid_results={}
    # Add the logits across the best checkpoints
    for test_outputs in sorted_predictions:
        for i in range(num_instances):
            combined_logits[i] += test_outputs[i]  # Element-wise addition
    
    # Determine the final class by selecting the max class for each instance
    final_predictions_5 = torch.tensor([np.argmax(logits) for logits in combined_logits])
    acc_5 = (final_predictions_5 == targets).sum().item()/targets.size(0)

              
    return acc_5



def test_meta(model, meta_loader,meta_goal,args):
    model.eval()
    correct = 0
    test_loss = 0
    n=0
    features = []

    # with torch.no_grad():
    for batch_idx, batch in enumerate(meta_loader):
        if len(batch)==3:
            inputs, targets,_=batch
        else:
            inputs, targets = batch
        n_instances=len(targets)
        n+=n_instances
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs, feat = model(inputs,feat=True)
        _, predicted = outputs.max(1)

        if meta_goal =='clid':
            test_loss += clid_loss(feat,outputs, args.tau).item()*n_instances
            # +args.w_nege*neg_entropy(outputs).item()*n_instances
        elif meta_goal =='mae':
            test_loss += mae_loss(outputs, targets, reduction='mean').item()*n_instances
        elif meta_goal =='ce_sloss':
            test_loss += F.cross_entropy(outputs, targets).item()*n_instances
        elif meta_goal =='ce':
            test_loss += F.cross_entropy(outputs, targets).item()*n_instances
                

        correct += predicted.eq(targets).sum().item()
    test_loss /=n
    accuracy = 100. * correct / len(meta_loader.dataset)

    return accuracy, test_loss
    
def test_meta_s(model, meta_loader,meta_goal,args):
    model.eval()
    correct = 0
    test_loss = 0
    n=0
    features = []
    with torch.no_grad():
        for batch_idx, (inputs,_,targets,_) in enumerate(meta_loader):
            n_instances=len(targets)
            n+=n_instances
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, feat = model(inputs,feat=True)
            _, predicted = outputs.max(1)

            if meta_goal =='clid':
                test_loss += clid_loss(feat,outputs, args.tau).item()*n_instances
                # +args.w_nege*neg_entropy(outputs).item()*n_instances
            elif meta_goal =='mae':
                test_loss += mae_loss(outputs, targets, reduction='mean').item()*n_instances
            elif meta_goal =='ce_sloss':
                test_loss += F.cross_entropy(outputs, targets).item()*n_instances
            elif meta_goal =='ce':
                test_loss += F.cross_entropy(outputs, targets).item()*n_instances
                    

            correct += predicted.eq(targets).sum().item()
    test_loss /=n
    accuracy = 100. * correct / len(meta_loader.dataset)

    return accuracy, test_loss

def test_m(model, test_loader):
    model.eval()
    correct = 0
    test_loss = 0
    outputs_all = []
    targets_all =[]
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            # print('len_targets',len(targets),targets)
            # if targets.shape[0]!=1:
            #     targets=targets[0]
            inputs, targets = inputs.cuda(), targets.cuda()
            # n_instances=len(targets)
            outputs,_ = model(inputs,feat=True)
            # print('outputs',outputs)
            # test_loss += F.cross_entropy(outputs, targets).item()*n_instances
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            outputs_all.append(outputs.detach().cpu().numpy())
            targets_all.append(targets.cpu())
    # test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    return accuracy, np.concatenate(outputs_all,axis=0), torch.cat(targets_all, dim=0)

def test_m_food(model, test_loader):
    model.eval()
    correct = 0
    test_loss = 0
    outputs_all = []
    targets_all =[]
    with torch.no_grad():
        for batch_idx, (inputs, targets,_) in enumerate(test_loader):

            inputs, targets = inputs.cuda(), targets.cuda()
            n_instances=len(targets)
            outputs,_ = model(inputs,feat=True)
            test_loss += F.cross_entropy(outputs, targets).item()*n_instances
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            outputs_all.append(outputs.detach().cpu().numpy())
            targets_all.append(targets.cpu())
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    return accuracy, np.concatenate(outputs_all,axis=0), torch.cat(targets_all, dim=0)

def loss_function(logits, onehot_label):
    log_prob = torch.nn.functional.log_softmax(logits, dim=1)
    loss = - torch.sum(log_prob * onehot_label) / logits.size(0)
    return loss
    
def kl_loss(mu1, log_var1, mu2, log_var2):
    s = (log_var2 - log_var1) + (torch.pow(torch.exp(log_var1), 2) + torch.pow(mu1 - mu2, 2)) / (
                2 * torch.pow(torch.exp(log_var2), 2)) - 0.5
    return torch.mean(s)





def clid_loss(feat,logits,contrast_th=0.1):
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

def mae_loss(logits, targets, reduction='mean'):
    num_classes =  logits.shape[1]
    pred = F.softmax(logits, dim=-1)
    if logits.shape != targets.shape:
        # conver to one-hot target
        targets = torch.nn.functional.one_hot(targets, num_classes).float().to(logits.device)
    
    mae_loss = 1.0 - torch.sum(targets * pred, dim=1)
    
    if reduction == 'none':
        return mae_loss
    else:
        return mae_loss.mean()