import torch
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture
import random
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import numpy as np
import torch.nn as nn
import copy
from dataloader import CustomSubset

class EMA:
    """
    EMA model
    Implementation from https://fyubang.com/2019/06/01/ema/
    """

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def load(self, ema_model):
        for name, param in ema_model.named_params(ema_model):
            self.shadow[name] = param.data.clone()

    def register(self):
        for name, param in self.model.named_params(self.model):
            self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_params(self.model):
            new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
            self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_params(self.model):
            self.backup[name] = param.data
            param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_params(self.model):
            param.data = self.backup[name]
        self.backup = {}




def neg_entropy(outputs):
    probs = torch.softmax(outputs, dim=1)
    return torch.mean(torch.sum(probs.log()*probs, dim=1))

def kl_loss(mu1, log_var1, mu2, log_var2):
    s = (log_var2 - log_var1) + (torch.pow(torch.exp(log_var1), 2) + torch.pow(mu1 - mu2, 2)) / (
                2 * torch.pow(torch.exp(log_var2), 2)) - 0.5
    return torch.mean(s)

def lid(batch, k=20, distmetric='l2') -> torch.Tensor:
    k = min(k, batch.shape[0] - 1)
    if distmetric == 'cos':
        a_norm = batch / batch.norm(p=2, dim=1)[:, None]

        # cosine distance: 1-cos()
        cos_sim = torch.mm(a_norm, a_norm.transpose(0, 1))
        cos_distance = torch.ones(cos_sim.size()).to(batch.device) - cos_sim
        distance_sorted, indices = torch.sort(cos_distance)
    else:
        assert distmetric == 'l2'
        distance = torch.cdist(batch, batch, p=2)
        distance_sorted, indices = torch.sort(distance)

    selected = distance_sorted[:, 1:k + 1] + 1e-12
    lids_log_term = torch.sum(torch.log(selected / (selected[:, -1]).reshape(-1, 1)), dim=1)
    lids_log_term = lids_log_term + 1e-12
    lids = -k / lids_log_term
    return lids.mean()
    
def svd_loss(mlp_feats):
    u, s, v = torch.svd(mlp_feats)
    s = s / torch.sum(s)
    # loss -= svd_loss_weight * torch.mean(s[:5])
    return -torch.mean(s[:5])

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
def sve(features):
    svd = TruncatedSVD(n_components=features.shape[1])
    svd.fit(features)
    components = svd.components_
    # print(u_vectors.shape)
    singular_values = svd.singular_values_
    P = singular_values / np.sum(singular_values) # Target distribution
    sve = -np.sum(P * np.log(P + 1e-12))
    return sve    
    
def cov_loss(mlp_feats):
    mlp_feats = mlp_feats - mlp_feats.mean(dim=0)
    cov_mlp_feats = (mlp_feats.T @ mlp_feats) / (mlp_feats.size(0) - 1)
    cov_loss = off_diagonal(cov_mlp_feats).pow_(2).sum().div(mlp_feats.size(1))
    return cov_loss         
    
def clid_loss(feat,logits,temperature=0.5):
    
    probs_x_ulb = torch.softmax(logits, dim=-1)
    max_probs, y = torch.max(probs_x_ulb, dim=-1)
    feat = F.normalize(feat)
    normalized_probs = F.normalize(probs_x_ulb) #l2norm??
    
    m_feat = feat.shape[1] # m_feat is the # of dimensions of the features
    n_feat = feat.shape[0] # n_feat is the # of instances 
    sim_n = torch.mm(feat, feat.t())/temperature 

    pos_mask = (sim_n>=0).float()
    sim_n = sim_n * pos_mask
    s_ij1 = F.softmax(sim_n,dim=-1)
    
    ####
    eps= 1e-7
    weight_graph = torch.mm(normalized_probs,normalized_probs.t())
    
    weight_graph_ = weight_graph/weight_graph.sum(1,keepdim=True).detach()

    assert weight_graph.shape==(n_feat,n_feat)

    entr = torch.sum(-torch.log(s_ij1.detach()+eps)*weight_graph_,dim=1)

    return entr.mean()
    


# https://github.com/Virusdoll/Active-Negative-Loss/blob/main/loss.py
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





def loss_function(logits, onehot_label):
    log_prob = torch.nn.functional.log_softmax(logits, dim=1)
    loss = - torch.sum(log_prob * onehot_label) / logits.size(0)
    return loss

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def test(model, test_loader, loss=False):
    model.eval()
    correct = 0
    test_loss = 0
    outputs_all = []
    targets_all =[]
    n=0
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, _ = model(inputs)
            n_instances = inputs.shape[0]
            n+=n_instances
            _, predicted = outputs.max(1)
            test_loss += F.cross_entropy(outputs, targets,reduction='sum').item()

            correct += predicted.eq(targets).sum().item()
            outputs_all.append(outputs.detach().cpu().numpy())
            targets_all.append(targets.cpu())
    test_loss /= n
    accuracy = 100. * correct / n
    if loss:
        return accuracy, np.concatenate(outputs_all,axis=0), torch.cat(targets_all, dim=0), test_loss

    else:
        return accuracy, np.concatenate(outputs_all,axis=0), torch.cat(targets_all, dim=0)



def evaluate_model(model, test_loader):
    """
    Evaluates the model on a test dataset.
    
    Args:
        model (torch.nn.Module): The trained model.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (torch.device): The device (CPU or GPU) on which to run the evaluation.
    
    Returns:
        float: Average test loss.
        float: Test accuracy.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0
    device='cuda'
    with torch.no_grad():  # No gradient computation during evaluation
        for inputs, targets, _ in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs,_ = model(inputs)
            
            # Compute loss
            loss = F.cross_entropy(outputs, targets, reduction='sum')  # Sum loss over batch
            total_loss += loss.item()
            
            # Compute accuracy
            _, predicted = torch.max(outputs, dim=1)  # Get the predicted class
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    # Compute average loss and accuracy
    avg_loss = total_loss / total  # Average loss over dataset
    accuracy = 100 * correct / total  # Accuracy in percentage

    # print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy



def test_meta(model, meta_loader,meta_goal,args):
    model.eval()
    correct = 0
    test_loss = 0
    n=0
    features = []

    with torch.no_grad():
        for batch_idx, (inputs,inputs_s1,inputs_s2, targets, _,true_labels) in enumerate(meta_loader):
            n_instances=len(targets)
            n+=n_instances
            inputs, targets,true_labels = inputs.cuda(), targets.cuda(),true_labels.cuda()
            outputs, feat = model(inputs)
            _, predicted = outputs.max(1)

            if meta_goal =='clid':
                test_loss += clid_loss(feat,outputs, args.tau).item()*n_instances 

            elif meta_goal =='mae':
                test_loss += mae_loss(outputs, targets, reduction='mean').item()*n_instances
            elif meta_goal =='ce_sloss':
                test_loss += F.cross_entropy(outputs, targets).item()*n_instances
            elif meta_goal =='ce':
                test_loss += F.cross_entropy(outputs, true_labels).item()*n_instances
                    

            correct += predicted.eq(true_labels).sum().item()
            features.append(feat.cpu())
    features = torch.cat(features)
    lid_score =lid(features)
    lsvr = svd_loss(features).item()
    
    cov = cov_loss(features).item()
    features = torch.nn.functional.normalize(features, dim=1)
    features = features - features.mean(dim=1, keepdim=True)
    features = features.numpy()
    # run svd
    svd = TruncatedSVD(n_components=features.shape[1])
    svd.fit(features)
    components = svd.components_
    # print(u_vectors.shape)
    singular_values = svd.singular_values_
    P = singular_values / np.sum(singular_values) # Target distribution
    sve = -np.sum(P * np.log(P + 1e-12))

    test_loss /=n
    accuracy = 100. * correct / len(meta_loader.dataset)

    return accuracy, test_loss,sve, lid_score.item(),lsvr,cov

def eval_train(model, train_loader):
    num_meta=1000
    model.eval()
    data_ids=[]
    all_loss =[]
    y_train = []
    y_true = []
    train_set = train_loader.dataset
    with torch.no_grad():
        for batch_idx, (inputs,_,_, targets, ids ,true_labels) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, feat = model(inputs)
            loss = F.cross_entropy(outputs, targets,reduction='none')
            y_train.extend(targets.cpu().tolist())
            y_true.extend(true_labels.tolist())
            all_loss.extend(loss.cpu().numpy())
            data_ids.extend(ids.cpu().numpy().tolist())
    
    num_classes=outputs.shape[1]        
    data_info = pd.DataFrame({'dataset_ids':data_ids,'label': y_train,'loss': all_loss,'true_label':y_true})
    # Step 3: Select top samples with smallest losses for each class
    selected_indices = []
    samples_per_class = num_meta // num_classes  # Assuming 10 classes
    data_info['clean']= (data_info['label']==data_info['true_label']).astype(int)

    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    input_loss= data_info['loss'].values.reshape(-1, 1)
    clean =data_info['clean'].values
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
    noise_ratio = 1-(selected_data_info['label']==selected_data_info['true_label']).mean()
    random.shuffle(selected_indices)
    meta_dset = CustomSubset(train_set,selected_indices)
    reset_subset = CustomSubset(meta_dset, range(len(meta_dset)))

    return noise_ratio, reset_subset




def update_best_results_dict(best_checkpoints,meta_clid, outputs,epoch, targets,max_size=5):
    # Add the new checkpoint with its metric and epoch as the key
    best_checkpoints[epoch] = {'meta_clid': meta_clid, 'test_outputs': outputs}
    
    # Sort by metric (keep the top 5)
    sorted_checkpoints = sorted(best_checkpoints.items(), key=lambda x: x[1]['meta_clid'], reverse=False)  # Use reverse=False for loss
    best_checkpoints_5 = dict(sorted_checkpoints[:max_size])
    best_checkpoints_3 = dict(sorted_checkpoints[:3])
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
    for checkpoint in best_checkpoints_3.values():
        test_outputs = checkpoint['test_outputs']
        for i in range(num_instances):
            combined_logits[i] += test_outputs[i]  # Element-wise addition
    
    # Determine the final class by selecting the max class for each instance
    final_predictions_3 = torch.tensor([np.argmax(logits) for logits in combined_logits])
    acc_3 = (final_predictions_3 == targets).sum().item()/targets.size(0)
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
    meta_clid_results['acc_3']=acc_3
    meta_clid_results['acc_1']=acc_1
              
    return meta_clid_results, best_checkpoints_5


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
            inputs, targets = inputs.cuda(), targets.cuda()
            n_instances=len(targets)
            outputs = model(inputs)
            test_loss += F.cross_entropy(outputs, targets).item()*n_instances
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            outputs_all.append(outputs.detach().cpu().numpy())
            targets_all.append(targets.cpu())
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

