import torch
import torch.nn as nn
import config

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from models.base_model import TrashBaseClass


class TrashResnet(TrashBaseClass):
    transfer_learning = True

    def __init__(self, resnet_model, *args, **kwargs):
        super().__init__()
        self.resnet_model = resnet_model
        model_list = [resnet18, resnet34, resnet50, resnet101, resnet152]
        backbone = model_list[self.resnet_model](pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.backbone = nn.Sequential(*layers)

        num_target_classes = config.NUM_CLASSES
        self.classifier = nn.Linear(num_filters, num_target_classes)

    def forward(self, x):
        x = self.backbone(x).flatten(1)
        x = self.classifier(x)
        return x


class TrashResnet18(TrashResnet):
    def __init__(self, *args, **kwargs):
        super().__init__(0, *args, **kwargs)


class TrashResnet34(TrashResnet):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)


class TrashResnet50(TrashResnet):
    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)


class TrashResnet101(TrashResnet):
    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)


class TrashResnet152(TrashResnet):
    def __init__(self, *args, **kwargs):
        super().__init__(4, *args, **kwargs)
