import torch.nn as nn
import torch
from torchvision import models

class sinkhorn(nn.Module):
    def __init__(self, n_iter=3):
        super(sinkhorn, self).__init__()
        self.n_iter = n_iter

    def forward(self, C, epsilon):
        batch_size, n, m = C.shape
        u = torch.ones(batch_size, n, 1).to(C.device)
        v = torch.ones(batch_size, 1, m).to(C.device)
        for i in range(self.n_iter):
            u = 1.0 / (torch.matmul(C, v) + epsilon) * u
            v = 1.0 / (torch.matmul(u.transpose(1, 2), C) + epsilon) * v
        return u, v

class Backbone(nn.Module):

    def __init__(self, backbone='resnet50', input_channel=3, pretrained=True):
        super(Backbone, self).__init__()
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
        elif backbone == 'resnet152':
            self.backbone = models.resnet152(pretrained=pretrained)
        elif backbone == 'dense121':
            self.backbone = models.densenet121(pretrained=pretrained)
        elif backbone == 'dense161':
            self.backbone = models.densenet161(pretrained=pretrained)
        elif backbone == 'dense169':
            self.backbone = models.densenet169(pretrained=pretrained)
        self.backbone.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    def freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.backbone(x)
        return x