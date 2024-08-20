import torch
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, PrecisionRecallDisplay


def parse_tensorboard(path, scalars):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    scalars = ea.Tags()["scalars"]
    # make sure the scalars are in the event accumulator tags
    assert all(
        s in ea.Tags()["scalars"] for s in scalars
    ), "some scalars were not found in the event accumulator"
    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}


def cal_metrics(log_, suffix,var_name,m=None):
    """var_name: ce, logits, feat"""

    if suffix=='lid_20':
        if m!='cos':
            pre_feat = log_['lid_20/tp_{}'.format(var_name)]['value'] / (log_['lid_20/tp_{}'.format(var_name)]['value']+log_['lid_20/fp_{}'.format(var_name)]['value'])
            recall_feat = log_['lid_20/tp_{}'.format(var_name)]['value'] / (log_['lid_20/tp_{}'.format(var_name)]['value']+log_['lid_20/fn_{}'.format(var_name)]['value'])
            f1_feat = 2*pre_feat*recall_feat / (pre_feat + recall_feat) 
        else:
            pre_feat = log_['lid_20/tp_{}_cos'.format(var_name)]['value'] / (log_['lid_20/tp_{}_cos'.format(var_name)]['value']+log_['lid_20/fp_{}_cos'.format(var_name)]['value'])
            recall_feat = log_['lid_20/tp_{}_cos'.format(var_name)]['value'] / (log_['lid_20/tp_{}_cos'.format(var_name)]['value']+log_['lid_20/fn_{}_cos'.format(var_name)]['value'])
            f1_feat = 2*pre_feat*recall_feat / (pre_feat + recall_feat) 

    elif suffix =='lid_40':
        if m!='cos':
            pre_feat = log_['lid_40/tp_{}'.format(var_name)]['value'] / (log_['lid_40/tp_{}'.format(var_name)]['value']+log_['lid_40/fp_{}'.format(var_name)]['value'])
            recall_feat = log_['lid_40/tp_{}'.format(var_name)]['value'] / (log_['lid_40/tp_{}'.format(var_name)]['value']+log_['lid_40/fn_{}'.format(var_name)]['value'])
            f1_feat = 2*pre_feat*recall_feat / (pre_feat + recall_feat) 
        else:
            pre_feat = log_['lid_40/tp_{}_cos'.format(var_name)]['value'] / (log_['lid_40/tp_{}_cos'.format(var_name)]['value']+log_['lid_40/fp_{}_cos'.format(var_name)]['value'])
            recall_feat = log_['lid_40/tp_{}_cos'.format(var_name)]['value'] / (log_['lid_40/tp_{}_cos'.format(var_name)]['value']+log_['lid_40/fn_{}_cos'.format(var_name)]['value'])
            f1_feat = 2*pre_feat*recall_feat / (pre_feat + recall_feat) 
     
    
    else:
        if m!='cos':
            pre_feat = log_['lid_80/tp_{}'.format(var_name)]['value'] / (log_['lid_80/tp_{}'.format(var_name)]['value']+log_['lid_80/fp_{}'.format(var_name)]['value'])
            recall_feat = log_['lid_80/tp_{}'.format(var_name)]['value'] / (log_['lid_80/tp_{}'.format(var_name)]['value']+log_['lid_80/fn_{}'.format(var_name)]['value'])
            f1_feat = 2*pre_feat*recall_feat / (pre_feat + recall_feat) 
        
        
        else:
            pre_feat = log_['lid_80/tp_{}_cos'.format(var_name)]['value'] / (log_['lid_80/tp_{}_cos'.format(var_name)]['value']+log_['lid_80/fp_{}_cos'.format(var_name)]['value'])
            recall_feat = log_['lid_80/tp_{}_cos'.format(var_name)]['value'] / (log_['lid_80/tp_{}_cos'.format(var_name)]['value']+log_['lid_80/fn_{}_cos'.format(var_name)]['value'])
            f1_feat = 2*pre_feat*recall_feat / (pre_feat + recall_feat) 
            
    return pre_feat, recall_feat, f1_feat



def cal_metrics_r(log_, suffix,var_name,m=None):
    """var_name: ce, logits, feat"""

    if suffix=='lid_20':
        if m!='cos':
            pre_feat = log_['lid_20/tn_{}'.format(var_name)]['value'] / (log_['lid_20/tn_{}'.format(var_name)]['value']+log_['lid_20/fn_{}'.format(var_name)]['value'])
            recall_feat = log_['lid_20/tn_{}'.format(var_name)]['value'] / (log_['lid_20/tn_{}'.format(var_name)]['value']+log_['lid_20/fp_{}'.format(var_name)]['value'])
            f1_feat = 2*pre_feat*recall_feat / (pre_feat + recall_feat) 
        else:
            pre_feat = log_['lid_20/tn_{}_cos'.format(var_name)]['value'] / (log_['lid_20/tn_{}_cos'.format(var_name)]['value']+log_['lid_20/fn_{}_cos'.format(var_name)]['value'])
            recall_feat = log_['lid_20/tn_{}_cos'.format(var_name)]['value'] / (log_['lid_20/tn_{}_cos'.format(var_name)]['value']+log_['lid_20/fp_{}_cos'.format(var_name)]['value'])
            f1_feat = 2*pre_feat*recall_feat / (pre_feat + recall_feat) 

    elif suffix =='lid_40':
        if m!='cos':
            pre_feat = log_['lid_40/tn_{}'.format(var_name)]['value'] / (log_['lid_40/tn_{}'.format(var_name)]['value']+log_['lid_40/fn_{}'.format(var_name)]['value'])
            recall_feat = log_['lid_40/tn_{}'.format(var_name)]['value'] / (log_['lid_40/tn_{}'.format(var_name)]['value']+log_['lid_40/fp_{}'.format(var_name)]['value'])
            f1_feat = 2*pre_feat*recall_feat / (pre_feat + recall_feat) 
        else:
            pre_feat = log_['lid_40/tn_{}_cos'.format(var_name)]['value'] / (log_['lid_40/tn_{}_cos'.format(var_name)]['value']+log_['lid_40/fn_{}_cos'.format(var_name)]['value'])
            recall_feat = log_['lid_40/tn_{}_cos'.format(var_name)]['value'] / (log_['lid_40/tn_{}_cos'.format(var_name)]['value']+log_['lid_40/fp_{}_cos'.format(var_name)]['value'])
            f1_feat = 2*pre_feat*recall_feat / (pre_feat + recall_feat) 
     
    
    else:
        if m!='cos':
            pre_feat = log_['lid_80/tn_{}'.format(var_name)]['value'] / (log_['lid_80/tn_{}'.format(var_name)]['value']+log_['lid_80/fn_{}'.format(var_name)]['value'])
            recall_feat = log_['lid_80/tn_{}'.format(var_name)]['value'] / (log_['lid_80/tn_{}'.format(var_name)]['value']+log_['lid_80/fp_{}'.format(var_name)]['value'])
            f1_feat = 2*pre_feat*recall_feat / (pre_feat + recall_feat) 
        
        
        else:
            pre_feat = log_['lid_80/tn_{}_cos'.format(var_name)]['value'] / (log_['lid_80/tn_{}_cos'.format(var_name)]['value']+log_['lid_80/fn_{}_cos'.format(var_name)]['value'])
            recall_feat = log_['lid_80/tn_{}_cos'.format(var_name)]['value'] / (log_['lid_80/tn_{}_cos'.format(var_name)]['value']+log_['lid_80/fp_{}_cos'.format(var_name)]['value'])
            f1_feat = 2*pre_feat*recall_feat / (pre_feat + recall_feat) 
            
    return pre_feat, recall_feat, f1_feat



# def cal_metrics_r(log_, var_name):
#     """cetrics for clean lables; var_name: ce, logits, feat"""
    
#     pre_feat = log_['tn_{}'.format(var_name)]['value'] / (log_['tn_{}'.format(var_name)]['value']+log_['fn_{}'.format(var_name)]['value'])
#     recall_feat = log_['tn_{}'.format(var_name)]['value'] / (log_['tn_{}'.format(var_name)]['value']+log_['fp_{}'.format(var_name)]['value'])
#     f1_feat = 2*pre_feat*recall_feat / (pre_feat + recall_feat) 
#     return pre_feat, recall_feat, f1_feat


def clean_ratio_batch(log_,var_name):
    N = log_['lid_80/tn_{}'.format(var_name)]['value']+ log_['lid_80/fp_{}'.format(var_name)]['value']
    all_ = N + log_['lid_80/tp_{}'.format(var_name)]['value']+log_['lid_80/fn_{}'.format(var_name)]['value']
    return N/all_

# def cal_metrics_r(log_, var_name):
#     """cetrics for clean lables; var_name: ce, logits, feat"""
    
#     pre_feat = log_['tn_{}'.format(var_name)]['value'] / (log_['tn_{}'.format(var_name)]['value']+log_['fn_{}'.format(var_name)]['value'])
#     recall_feat = log_['tn_{}'.format(var_name)]['value'] / (log_['tn_{}'.format(var_name)]['value']+log_['fp_{}'.format(var_name)]['value'])
#     f1_feat = 2*pre_feat*recall_feat / (pre_feat + recall_feat) 
#     return pre_feat, recall_feat, f1_feat


# def clean_ratio_batch(log_,var_name):
#     N = log_['tn_{}'.format(var_name)]['value']+ log_['fp_{}'.format(var_name)]['value']
#     all_ = N + log_['tp_{}'.format(var_name)]['value']+log_['fn_{}'.format(var_name)]['value']
#     return N/all_
    


def mean_(data):
    n = len(data)
    mean = sum(data) / n
    return mean

def std_(data):
    n = len(data)
    mean = sum(data) / n
    var = sum((x - mean)**2 for x in data) / n
    std = var ** 0.5
    return std


"""https://github.com/rachellea/glassboxmedicine/blob/master/2020-07-14-AUROC-AP/main.py"""
def confusion_matrix_values(y_true, y_score, decision_thresh):
    #Obtain binary predicted labels by applying <decision_thresh> to <y_score>
    y_pred = (np.array(y_score) > decision_thresh)
    cm = sklearn.metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
    true_neg, false_pos, false_neg, true_pos = cm.ravel()
    return true_neg, false_pos, false_neg, true_pos

def calculate_tpr_fpr_prec(y_true, y_score, decision_thresh):
    true_neg, false_pos, false_neg, true_pos =  confusion_matrix_values(y_true, y_score, decision_thresh)
    tpr_recall = float(true_pos)/(true_pos + false_neg)
    fpr = float(false_pos)/(false_pos+true_neg)
    precision = float(true_pos)/(true_pos + false_pos)
    return tpr_recall, fpr, precision

def cal_auprc(noise_weight,clean_weight,i):
    m = torch.nn.Sigmoid()
    output_n = m(torch.Tensor(noise_weight[i]))
    output_c = m(torch.Tensor(clean_weight[i]))
    
    y_true = [0]*len(output_n)+[1]*len(output_c)
    pred = torch.cat((output_n,output_c))
    auprc = average_precision_score( y_true, pred )
    precision, recall, thre = precision_recall_curve(y_true, pred)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    for d in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8, 0.9]:
        tpr_recall, _, precision = calculate_tpr_fpr_prec(y_true, pred, d)
        plt.plot(tpr_recall, precision, 'o', color = 'r')
        text = plt.annotate('d='+str(d), (tpr_recall, precision))
        text.set_rotation(45)
    plt.show()
    print('AUPRC:',auprc)
    return auprc


def get_g_clean_noisy(weight_log,key):
    noise_g_logits=[]

    clean_g_logits=[]
    for i,batch_idx in enumerate(weight_log['sample_idx']):
        noise_g_logits.append([weight_log[key][i][j].item() for j,idx_ in enumerate(batch_idx) if idx_.item() not in weight_log['clean_indices']])

        clean_g_logits.append([weight_log[key][i][j].item() for j,idx_ in enumerate(batch_idx) if idx_.item() in weight_log['clean_indices'] ])
    return noise_g_logits,clean_g_logits


