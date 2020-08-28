import torch.nn as nn
from torchvision import models

class Resnet_A(nn.Module):

    def __init__(self, num_classes, layers, u_dim=64):
        super(Resnet_A, self).__init__()
        if layers == 18:
            self.net = models.resnet18(pretrained=False, num_classes=u_dim)
        elif layers == 50:
            self.net = models.resnet50(pretrained=False, num_classes=u_dim)
        elif layers == 101:
            self.net = models.resnet101(pretrained=False, num_classes=u_dim)
        else:
            raise ValueError("Wrong number of layers for resnet")
        self.classifier = nn.Linear(u_dim, num_classes)

    def forward(self, input):
        out = self.net(input)
        logits = self.classifier(out)
        return logits