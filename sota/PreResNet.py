import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        # self.feat=feat
        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, lin=0, lout=5,feat=False):
        out = x
        if lin < 1 and lout > -1:
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)
        if lin < 2 and lout > 0:
            out = self.layer1(out)
        if lin < 3 and lout > 1:
            out = self.layer2(out)
        if lin < 4 and lout > 2:
            out = self.layer3(out)
        if lin < 5 and lout > 3:
            out = self.layer4(out)
        if lout > 4:
            out = F.avg_pool2d(out, 4)
            feat_e = out.view(out.size(0), -1)
            out = self.linear(feat_e)
        if feat:
            return out, feat_e
        else:
            return out

def ResNet18(num_classes=10):
    return ResNet(PreActBlock, [2,2,2,2], num_classes=num_classes)

def ResNet34(pretrained=False,num_classes=10):
    model = ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes)

def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes)

def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes)


def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())


class vri(nn.Module):
    def __init__(self, input, hidden1, hidden2, output):
        # 64, 128, 10
        super(vri, self).__init__()

        self.linear1 = nn.Linear(input, hidden1)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.linear3 = nn.Linear(hidden2, hidden2)
        self.linear_mean = nn.Linear(hidden2, output)
        self.linear_var = nn.Linear(hidden2, output)

        self.cls_emb = nn.Embedding(output, 64)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_normal_(self.linear1.weight)
        self.linear1.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.linear2.weight)
        self.linear2.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.linear3.weight)
        self.linear3.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.linear_mean.weight)
        self.linear_mean.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.linear_var.weight)
        self.linear_var.bias.data.zero_()

    def encode(self, x):
        h1 = self.tanh(self.linear1(x))
        h2 = self.tanh(self.linear2(h1))
        h3 = self.tanh(self.linear3(h2))
        mean = self.linear_mean(h3)
        std = self.linear_var(h3)
        return mean, std

    def forward(self, dict,one=False):
        # print('feat-shape:',dict['feature'].shape)
        if 'target' in dict:
            if not one:
                target = torch.argmax(dict['target'], dim=1)
                target = self.cls_emb(target.long())
            else:
                target = self.cls_emb(dict['target'])

            x = torch.cat([dict['feature'], target], dim=-1)
        else:
            x = torch.cat([dict['feature'], torch.zeros(dict['feature'].shape[0], 64).cuda()], dim=-1)
        mean, log_var = self.encode(x)  # [100, 10]
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        # s = 1 + 2 * torch.log(std) - mean.pow(2) - std.pow(2)
        # kl_loss = -0.5 * torch.mean(s)

        # return mean, std, F.sigmoid(mean + std*eps), torch.norm(std,2,dim=1).mean()
        return mean, std, F.sigmoid(mean + std * eps) #, torch.max(std, dim=0).values.mean()

class vri_mix(nn.Module):
    def __init__(self, input, hidden1, hidden2, output):
        # 64, 128, 10
        super(vri_mix, self).__init__()

        self.linear1 = nn.Linear(input, hidden1)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.linear3 = nn.Linear(hidden2, hidden2)
        self.linear_mean = nn.Linear(hidden2, output)
        self.linear_var = nn.Linear(hidden2, output)

        self.cls_emb = nn.Linear(output, 64)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_normal_(self.linear1.weight)
        self.linear1.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.linear2.weight)
        self.linear2.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.linear3.weight)
        self.linear3.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.linear_mean.weight)
        self.linear_mean.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.linear_var.weight)
        self.linear_var.bias.data.zero_()

    def encode(self, x):
        h1 = self.tanh(self.linear1(x))
        h2 = self.tanh(self.linear2(h1))
        h3 = self.tanh(self.linear3(h2))
        mean = self.linear_mean(h3)
        std = self.linear_var(h3)
        return mean, std

    def forward(self, dict):
        # print('feat-shape:',dict['feature'].shape)
        if 'target' in dict:
            # target = torch.argmax(dict['target'], dim=1)
            target = self.cls_emb(dict['target'])
            x = torch.cat([dict['feature'], target], dim=-1)
        else:
            x = torch.cat([dict['feature'], torch.zeros(dict['feature'].shape[0], 64).cuda()], dim=-1)
        mean, log_var = self.encode(x)  # [100, 10]
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        # s = 1 + 2 * torch.log(std) - mean.pow(2) - std.pow(2)
        # kl_loss = -0.5 * torch.mean(s)

        # return mean, std, F.sigmoid(mean + std*eps), torch.norm(std,2,dim=1).mean()
        return mean, std, F.sigmoid(mean + std * eps) #, torch.max(std, dim=0).values.mean()

class VNet(nn.Module):
    def __init__(self, input, hidden1, output):
        super(VNet, self).__init__()
        self.linear1 = nn.Linear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden1, output)
        # self.linear3 = MetaLinear(hidden2, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        # x = self.linear2(x)
        # x = self.relu1(x)
        out = self.linear2(x)
        return torch.sigmoid(out)

