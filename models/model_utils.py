import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class GDWSResnet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, temp=1.0):
        super(GDWSResnet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.temp = temp
        self.PF_epochs = []

        self.arch_parameters = nn.Parameter(
            1e-2 * torch.randn(5, 7)
        )
        self.tau = 10

        self.arch_choice = [50, 100, 150, 200, 250, 300, 350]

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def set_tau(self, tau):
        self.tau = tau

    def get_tau(self):
        return self.tau

    def get_alphas(self):
        return [self.arch_parameters]

    def get_arch(self):
        return (torch.argmax(nn.functional.softmax(self.arch_parameters, dim=-1).cpu(), 1) + 1) * 50

    def show_alphas(self):
        with torch.no_grad():
            return "arch-parameters :\n{:}".format(
                nn.functional.softmax(self.arch_parameters, dim=-1).cpu()
            )

    def load_baseline(self):
        self.baseline_50 = gwdsresnet50(pretrain=False).cuda()
        self.baseline_100 = gwdsresnet50(pretrain=False).cuda()
        self.baseline_150 = gwdsresnet50(pretrain=False).cuda()
        self.baseline_200 = gwdsresnet50(pretrain=False).cuda()
        self.baseline_250 = gwdsresnet50(pretrain=False).cuda()
        self.baseline_300 = gwdsresnet50(pretrain=False).cuda()
        self.baseline_350 = gwdsresnet50(pretrain=False).cuda()

        self.baseline_50.load_state_dict(torch.load(f"./weights/Vanilla_CE/cross_entropy_50.pt"),
                                         strict=False)
        self.baseline_100.load_state_dict(
            torch.load(f"./weights/Vanilla_CE/cross_entropy_100.pt"), strict=False)
        self.baseline_150.load_state_dict(
            torch.load(f"./weights/Vanilla_CE/cross_entropy_150.pt"), strict=False)
        self.baseline_200.load_state_dict(
            torch.load(f"./weights/Vanilla_CE/cross_entropy_200.pt"), strict=False)
        self.baseline_250.load_state_dict(
            torch.load(f"./weights/Vanilla_CE/cross_entropy_250.pt"), strict=False)
        self.baseline_300.load_state_dict(
            torch.load(f"./weights/Vanilla_CE/cross_entropy_300.pt"), strict=False)
        self.baseline_350.load_state_dict(
            torch.load(f"./weights/Vanilla_CE/cross_entropy_350.pt"), strict=False)

        for param in self.baseline_50.parameters():
            param.requires_grad = True
        for param in self.baseline_100.parameters():
            param.requires_grad = True
        for param in self.baseline_150.parameters():
            param.requires_grad = True
        for param in self.baseline_200.parameters():
            param.requires_grad = True
        for param in self.baseline_250.parameters():
            param.requires_grad = True
        for param in self.baseline_300.parameters():
            param.requires_grad = True
        for param in self.baseline_350.parameters():
            param.requires_grad = True

    def forward(self, inputs, arch=True):
        while True:
            gumbels = -torch.empty_like(self.arch_parameters).exponential_().log()
            logits = (self.arch_parameters.log_softmax(dim=1) + gumbels / 10) / self.tau
            probs = nn.functional.softmax(logits, dim=1)
            index = probs.max(-1, keepdim=True)[1]
            one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            hardwts = one_h - probs.detach() + probs
            if (
                    (torch.isinf(gumbels).any())
                    or (torch.isinf(probs).any())
                    or (torch.isnan(probs).any())
            ):
                continue
            else:
                break
        # print("Gumbel Index:", (index.cpu().numpy().flatten() + 1) * 50)
        index = index.cpu().numpy().flatten()
        # index = [6, 6, 6, 6, 6]

        out = F.relu(self.bn1(self.conv1(inputs)))

        model4layer1 = 'baseline_{}'.format(self.arch_choice[index[0]])
        out = getattr(self, model4layer1).layer1(out) * hardwts[0][index[0]]

        model4layer2 = 'baseline_{}'.format(self.arch_choice[index[1]])
        out = getattr(self, model4layer2).layer2(out) * hardwts[1][index[1]]

        model4layer3 = 'baseline_{}'.format(self.arch_choice[index[2]])
        out = getattr(self, model4layer3).layer3(out) * hardwts[2][index[2]]

        model4layer4 = 'baseline_{}'.format(self.arch_choice[index[3]])
        out = getattr(self, model4layer4).layer4(out) * hardwts[3][index[3]]

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        model4fc = 'baseline_{}'.format(self.arch_choice[index[4]])
        logits = getattr(self, model4fc).fc(out) * hardwts[4][index[4]] / self.temp

        return logits

    def load_back(self):
        self.load_state_dict(torch.load("./weights/Vanilla_CE/cross_entropy_350.pt"),
                             strict=False)

    def load_weight(self, arch=None):
        if arch == None:
            arch = self.get_arch().cpu().numpy()
        weights = torch.load(
            f"./weights/Vanilla_CE/cross_entropy_{arch[0]}.pt")
        weights_rep = {}
        for k in weights.keys():
            if k.startswith('layer1'):
                weights_rep[k[7:]] = weights[k]
        self.layer1.load_state_dict(weights_rep)

        weights = torch.load(
            f"./weights/Vanilla_CE/cross_entropy_{arch[1]}.pt")
        weights_rep = {}
        for k in weights.keys():
            if k.startswith('layer2'):
                weights_rep[k[7:]] = weights[k]
        self.layer2.load_state_dict(weights_rep)

        weights = torch.load(
            f"./weights/Vanilla_CE/cross_entropy_{arch[2]}.pt")
        weights_rep = {}
        for k in weights.keys():
            if k.startswith('layer3'):
                weights_rep[k[7:]] = weights[k]
        self.layer3.load_state_dict(weights_rep)

        weights = torch.load(
            f"./weights/Vanilla_CE/cross_entropy_{arch[3]}.pt")
        weights_rep = {}
        for k in weights.keys():
            if k.startswith('layer4'):
                weights_rep[k[7:]] = weights[k]
        self.layer4.load_state_dict(weights_rep)

        weights = torch.load(
            f"./weights/Vanilla_CE/cross_entropy_{arch[4]}.pt")
        weights_rep = {}
        for k in weights.keys():
            if k.startswith('fc'):
                weights_rep[k[3:]] = weights[k]
        self.fc.load_state_dict(weights_rep)

    def test_forward(self, inputs):
        out = F.relu(self.bn1(self.conv1(inputs)))

        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        logits = self.fc(out) / self.temp

        return logits


def gwdsresnet18(temp=1.0, **kwargs):
    model = GDWSResnet(BasicBlock, [2, 2, 2, 2], temp=temp, **kwargs)
    return model


def gwdsresnet34(temp=1.0, **kwargs):
    model = GDWSResnet(BasicBlock, [3, 4, 6, 3], temp=temp, **kwargs)
    return model


def gwdsresnet50(temp=1.0, pretrain=True, **kwargs):
    model = GDWSResnet(Bottleneck, [3, 4, 6, 3], temp=temp, **kwargs)

    if pretrain:
        model.load_state_dict(torch.load("./weights/Vanilla_CE/cross_entropy_350.pt"),
                              strict=False)

    for param in model.parameters():
        param.requires_grad = False
    model.arch_parameters.requires_grad = True
    return model


def gwdsresnet101(temp=1.0, **kwargs):
    model = GDWSResnet(Bottleneck, [3, 4, 23, 3], temp=temp, **kwargs)
    return model


def gwdsresnet110(temp=1.0, **kwargs):
    model = GDWSResnet(Bottleneck, [3, 4, 26, 3], temp=temp, **kwargs)
    return model


def gwdsresnet152(temp=1.0, **kwargs):
    model = GDWSResnet(Bottleneck, [3, 8, 36, 3], temp=temp, **kwargs)
    return model
