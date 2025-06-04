# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from semilearn.nets.utils import load_checkpoint

momentum = 0.001


def mish(x):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function (https://arxiv.org/abs/1908.08681)"""
    return x * torch.tanh(F.softplus(x))


class PSBatchNorm2d(nn.BatchNorm2d):
    """How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/2001.11216)"""

    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x):
        return super().forward(x) + self.alpha


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001, eps=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001, eps=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=True) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, first_stride, num_classes, depth=28, widen_factor=2, drop_rate=0.3, **kwargs):
        super(WideResNet, self).__init__()
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=True)
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, first_stride, drop_rate)
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001, eps=0.001)
        self.bn2 = nn.BatchNorm2d(channels[2], momentum=0.001, eps=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.fc = nn.Linear(channels[3], num_classes)
        self.channels = channels[3]
        self.channels_2 = channels[2]
        self.num_features = channels[3]

        # rot_classifier for Remix Match
        # self.is_remix = is_remix
        # if is_remix:
        #     self.rot_classifier = nn.Linear(self.channels, 4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
        """

        if only_fc:
            return self.fc(x)
        
        out = self.extract(x)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)
        # out,out_i = self.extract(x)
        # out = F.adaptive_avg_pool2d(out, 1)
        # out = out.view(-1, self.channels)

        # out_i = F.adaptive_avg_pool2d(out_i, 1)
        # out_i = out_i.view(-1, self.channels_2)
        
        if only_feat:
            return out
        
        output = self.fc(out)
        result_dict = {'logits':output, 'feat':out}
        return result_dict

    def extract(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        return out
        # out0 = self.conv1(x)
        # out1 = self.block1(out0)
        # out2 = self.block2(out1)
        # out3 = self.block3(out2)
        # out3 = self.relu(self.bn1(out3))
        # out2 = self.relu(self.bn2(out2))
        # return out3,out2

    def group_matcher(self, coarse=False, prefix=''):
        matcher = dict(stem=r'^{}conv1'.format(prefix), blocks=r'^{}block(\d+)'.format(prefix) if coarse else r'^{}block(\d+)\.layer.(\d+)'.format(prefix))
        return matcher

    def no_weight_decay(self):
        nwd = []
        for n, _ in self.named_parameters():
            if 'bn' in n or 'bias' in n:
                nwd.append(n)
        return nwd


def wrn_28_2(pretrained=False, pretrained_path=None, **kwargs):
    model = WideResNet(first_stride=1, depth=28, widen_factor=2, **kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model


def wrn_28_10(pretrained=False, pretrained_path=None, **kwargs):
    model = WideResNet(first_stride=1, depth=28, widen_factor=10, **kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model

def wrn_28_8(pretrained=False, pretrained_path=None, **kwargs):
    model = WideResNet(first_stride=1, depth=28, widen_factor=8, **kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model



class lenet(nn.Module):
    def __init__(self, num_classes):
        super(lenet, self).__init__()
    
        layers = []
        layers.append(nn.Conv2d(1, 6, kernel_size=5))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))

        layers.append(nn.Conv2d(6, 16, kernel_size=5))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        
        layers.append(nn.Conv2d(16, 120, kernel_size=5))
        layers.append(nn.ReLU())
        self.main = nn.Sequential(*layers)
        
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)
        
        
    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 120)
        ## classifier
        x = self.fc1(x)  
        feature = F.relu(x)
        logits = self.fc2(feature)
        
        result_dict = {'logits':logits, 'feat':feature}

        return result_dict


# class lenet(torch.nn.Module):          
#     def __init__(self, num_classes):
#         super(lenet, self).__init__()
#         # Convolution (In LeNet-5, 32x32 images are given as input. Hence padding of 2 is done below)
#         self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
#         self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
#         self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
#         self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2) 
#         self.fc1 = torch.nn.Linear(16*5*5, 120)   
#         self.fc2 = torch.nn.Linear(120, 84)
#         self.fc3 = torch.nn.Linear(84, num_classes)
        
#     def forward(self, x):
#         x = F.relu(self.conv1(x))  
#         x = self.max_pool_1(x) 
#         x = F.relu(self.conv2(x))
#         x = self.max_pool_2(x)
#         feature = x.view(-1, 16*5*5)
#         x = F.relu(self.fc1(feature))
#         x = F.relu(self.fc2(x))
#         logits = self.fc3(x)
#         result_dict = {'logits':logits, 'feat':feature}
#         return result_dict
        
        
#     def forward(self, x):
#         x = self.main(x)
#         x = x.view(-1, 120)
#         ## classifier
#         x = self.fc1(x)  
#         feature = F.relu(x)
#         logits = self.fc2(x)
        
#         result_dict = {'logits':logits, 'feat':feature}

#         return result_dict



class LogisticRegression(nn.Module):
    def __init__(self,num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(2, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.linear(x))
        feature = F.relu(self.linear2(x))
        logits = self.linear3(feature)

        result_dict = {'logits':logits, 'feat':feature}
        return result_dict

    
       
    
def LeNet(pretrained=False, pretrained_path=None, **kwargs):
    model =lenet(**kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model




def Lo(pretrained=False, pretrained_path=None, **kwargs):
    model =LogisticRegression(**kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model

class HiddenLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(HiddenLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))
        
class MLP(nn.Module):
    def __init__(self, hidden_size=100, num_layers=1):
        super(MLP, self).__init__()
        self.first_hidden_layer = HiddenLayer(1, hidden_size)
        self.rest_hidden_layers = nn.Sequential(*[HiddenLayer(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.first_hidden_layer(x)
        x = self.rest_hidden_layers(x)
        x = self.output_layer(x)
        return torch.sigmoid(x)
        
class WNet(nn.Module):
    def __init__(self, input_, hidden, output):
        super(WNet, self).__init__()
        self.linear1 = nn.Linear(input_, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        out = self.linear2(x)
        return torch.sigmoid(out) #weight [0,1]
        # return torch.relu()  # weight[0,1] remove the false-labeled samples

if __name__ == '__main__':
    model = wrn_28_2(pretrained=False, num_classes=10)
