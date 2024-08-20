from abc import abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoConfig

class BackBone(nn.Module):
    def __init__(self, num_classes, binary_mode=False):
        if binary_mode:
            assert num_classes == 2
            num_classes = 1
        self.num_classes = num_classes
        super(BackBone, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        return self.dummy_param.device

    def get_device(self):
        return self.dummy_param.device

    @abstractmethod
    def forward(self, batch: Dict, return_features: Optional[bool] = False):
        pass

class BERTBackBone(BackBone):
    def __init__(self, name='roberta-base',num_classes=2, fine_tune_layers=-1, binary_mode=False):
        super(BERTBackBone, self).__init__(num_classes=num_classes, binary_mode=binary_mode)
        self.model_name = name
        self.config = AutoConfig.from_pretrained(name, num_labels=num_classes, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(name, config=self.config)

        if fine_tune_layers >= 0:
            for param in self.model.base_model.embeddings.parameters(): param.requires_grad = False
            if fine_tune_layers > 0:
                n_layers = len(self.model.base_model.encoder.layer)
                for layer in self.model.base_model.encoder.layer[:n_layers - fine_tune_layers]:
                    for param in layer.parameters():
                        param.requires_grad = False

    @abstractmethod
    def forward(self, batch: Dict, return_features: Optional[bool] = False):
        pass

class Classificationroberta(BERTBackBone):
    """
    Bert with a MLP on top for text classification
    """

    def __init__(self, num_classes=2,name='roberta-base', fine_tune_layers=-1, max_tokens=128, binary_mode=False, **kwargs):
        super(Classificationroberta, self).__init__(num_classes=num_classes, name=name, fine_tune_layers=fine_tune_layers, binary_mode=binary_mode)

        self.dropout = nn.Dropout(p=0.1,inplace=False)
        self.classifier = nn.Linear(768, num_classes)
        self.max_tokens = max_tokens
        self.hidden_size = 768

    def forward(self, x, only_fc=False, only_feat=False, return_embed=False, **kwargs):  # inputs: [batch, t]
        if only_fc:
            logits = self.classifier(x)
            return logits
        
        out_dict = self.model(**x, output_hidden_states=True, return_dict=True)
        last_hidden = out_dict['last_hidden_state']
        # drop_hidden = self.dropout(last_hidden)
        # pooled_output = torch.mean(drop_hidden, 1)
        pooled_output = last_hidden[:, 0, :] 
        
        if only_feat:
            return pooled_output
        
        logits = self.classifier(pooled_output)
        result_dict = {'logits':logits, 'feat':pooled_output}

        if return_embed:
            result_dict['embed'] = out_dict['hidden_states'][0]
            
        return result_dict


def roberta_base(pretrained=True, pretrained_path=None, **kwargs):
    model = Classificationroberta(name='roberta-base', **kwargs)
    return model