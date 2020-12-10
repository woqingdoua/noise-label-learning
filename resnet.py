'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torchvision
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_ = nn.Linear(512 * block.expansion, num_classes)
        self.w = nn.Parameter(torch.randn(10, 3), requires_grad=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_(x)  # (128,10)
        return x

    def forward(self, x):
        return self._forward_impl(x)

    def update_w(self, w):

        self.w = nn.Parameter(w, requires_grad=False)


class Protype(nn.Module):

    def __init__(self, classnum=10, lr=0.001):
        super(Protype, self).__init__()

        self.t = nn.Sequential(nn.Linear(10, 10))
        self.w = nn.Parameter(torch.randn(classnum, 6))
        self.loss_fn = torch.nn.CosineEmbeddingLoss(margin=0.5)
        self.lr = 0.1
        self.loss_fn = SCELoss()

    def forward(self, x):

        x = self.t(x)
        x = F.softmax(x, dim=-1)
        x = x.unsqueeze(1)
        x = torch.einsum('ijk,kl->ijl', x, self.w)
        # pre = F.cosine_similarity(x.repeat(1,10,1),self.w.unsqueeze(0).repeat(len(x),1,1),dim=-1)
        pre = torch.sum((x - self.w) ** 2, dim=-1) ** 1 / 2
        pre = F.softmax(-pre, dim=-1)

        return pre, x.squeeze()

    def update_param(self, x, targets, t=False):

        # 判断预测值与真实值是否相等
        targets1 = F.softmax(x, dim=-1)
        x = self.t(x)
        x = F.softmax(x, dim=-1)
        x = x.unsqueeze(1)
        x = torch.einsum('ijk,kl->ijl', x, self.w)
        '''
        pre_v = F.cosine_similarity(x,self.w)
        pre = torch.argmin(torch.sum(pre_v,dim=-1),dim=-1)
        pre_one_hot = torch.zeros(len(pre),10).cuda().scatter_(1,pre.unsqueeze(-1),torch.tensor(1.).cuda())
        pre_cor_or_not = (pre==targets).float().unsqueeze(-1)

        #预测相等时的损失
        t_loss = torch.sum(pre_v * pre_one_hot.unsqueeze(-1).repeat(1,1,3),dim=1) #(n,3)
        t_loss = torch.sum(t_loss * pre_cor_or_not.repeat(1,3))/torch.sum(pre_cor_or_not)

        #预测不等时的损失
        pre_v_f = torch.sum(pre_v*pre_one_hot.unsqueeze(-1).expand_as(pre_v),dim=1)
        target_one_hot = torch.zeros(len(pre),10).cuda().scatter_(1,targets.unsqueeze(-1),torch.tensor(1.).cuda()).unsqueeze(-1).repeat(1,1,3)
        t =  torch.sum(pre_v*target_one_hot,dim=1)
        l = torch.sum(pre_v_f - t,dim=-1)
        f_loss = torch.exp(-l) * (1.-pre_cor_or_not.squeeze())
        f_loss = torch.sum(f_loss)/torch.sum(1.-pre_cor_or_not.squeeze())
        f_loss /= torch.mean(torch.sum(torch.abs(self.w),dim=-1))

        f_loss = torch.sum(F.softmax(-torch.sum(pre_v,dim=-1),dim=-1) * pre_one_hot,dim=-1)
        f_loss = torch.sum(f_loss * (1-pre_cor_or_not))/torch.sum(1-pre_cor_or_not)
        loss = t_loss + f_loss
        '''

        # 向量间相似度作为正则化项
        '''
        #norm = torch.sum(F.cosine_similarity(self.w.unsqueeze(0).repeat(10,1,1),self.w.unsqueeze(1).repeat(1,10,1),dim=-1))/2
        simi = F.cosine_similarity(x.repeat(1,10,1),self.w.unsqueeze(0).repeat(len(x),1,1),dim=-1)

        if t == True:
            targets = torch.argmax(simi,dim=-1)
            #targets = (targets1==targets)*targets1 + (targets1!=targets)*targets2

        simi = F.softmax(simi,dim=-1)
        loss = F.cross_entropy(simi,targets)

        '''

        # norm = torch.sum(torch.abs((self.w.unsqueeze(0).repeat(10,1,1)-self.w.unsqueeze(1).repeat(1,10,1))),dim=-1)
        # norm = torch.mean(torch.mean(norm,dim=-1))
        # norm = torch.where(norm>50.,torch.tensor(50.).cuda(),norm)

        simi = torch.sum((x - self.w) ** 2, dim=-1) ** 1 / 2
        simi = F.softmax(-simi, dim=-1)
        # norm3 = torch.mean(torch.sum(simi**2,dim=-1))
        # loss = torch.sum(targets1*-torch.log(simi/targets1),dim=-1) #- 0.1*norm3 + 0.1*norm2
        # loss = torch.mean(loss,dim=-1)
        # targets = torch.argmax(targets1,dim=-1)
        # targets = torch.argmax(targets1,dim=-1) if t == True else targets.long()

        loss = self.loss_fn(simi, targets,
                            torch.argmax(targets1, dim=-1))  # F.cross_entropy(simi,torch.argmax(targets1,dim=-1))
        # loss = F.cross_entropy(simi,targets)

        return loss

    def update(self, x, target):

        l = x - self.w
        p = torch.sum(torch.abs(l), dim=-1)
        pre = torch.argmin(p, dim=-1)
        acc = torch.sum(pre == target).float() / len(target)

        for i in range(1):

            mark = torch.where(pre == target, torch.tensor(1.).cuda(), torch.tensor(-1.).cuda()).unsqueeze(
                -1).unsqueeze(-1)
            pre_one_hot = torch.zeros(len(target), 10).cuda().scatter_(-1, pre.unsqueeze(-1), 1).unsqueeze(-1).repeat(1,
                                                                                                                      1,
                                                                                                                      3)
            stat = torch.sum(pre_one_hot, dim=0)
            stat = stat + (stat == 0).float()
            grad = torch.sum(l * pre_one_hot * mark, dim=0) / stat
            grad = torch.abs(torch.mean(l, dim=0)) * grad
            w = self.w + self.lr * F.tanh(grad)

            l1 = x - w
            p1 = torch.sum(torch.abs(l1), dim=-1)
            pre1 = torch.argmin(p1, dim=-1)
            acc1 = torch.sum(pre1 == target).float() / len(target)

            # print(f'{acc},{acc1}')

            if acc1 > acc:
                self.w = nn.Parameter(w)


def ResNet34(num_classes=10):
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    # model = torchvision.models.resnet34(pretrained=True)
    # model.fc = nn.Linear(model.fc.in_features, num_classes)
    # model_dict = model.state_dict()
    # pretrained_dict = torch.load('models/resnet34-333f7ec4.pth')
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    return model


def ResNet50(num_classes=14):
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # model = ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes)
    return model


def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)


class Gradient_truc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, w, x):
        ctx.save_for_backward(w, x)
        output = torch.matmul(x, w.permute(1, 0))
        return output

    @staticmethod
    def backward(ctx, grad_output):  # grad_output(128,10)
        w, x = ctx.saved_variables  # x:(128,512)
        grad_w = grad_output.unsqueeze(-1).repeat(1, 1, 10) * x.unsqueeze(1).repeat(1, 10, 1)
        grad_x = torch.matmul(grad_output, w)

        mean_grad_w = torch.mean(grad_w, dim=0)
        mark_v = torch.sum(torch.abs(grad_w - mean_grad_w), dim=1)
        mark, _ = torch.topk(mark_v, dim=0, k=len(mark_v))
        mark = mark[int(len(x) * 0.2)]
        l = mark_v - mark
        mark = torch.where(l < 0, torch.tensor(1.).cuda(), torch.tensor(0.1).cuda())
        mark = mark.unsqueeze(-1).repeat(1, 1, 10).permute(0, 2, 1)
        grad_w = mark * grad_w

        return grad_w, grad_x


class SCELoss(torch.nn.Module):
    def __init__(self, alpha=1., beta=1, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, pre_labels, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.beta * rce.mean() + self.alpha * ce

        return loss



