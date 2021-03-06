import torch.nn as nn
import config

from torchvision.models import convnext_tiny, convnext_small, convnext_base, convnext_large
from models.base_model import TrashBaseClass


class TrashConvnext(TrashBaseClass):
    transfer_learning = True

    def __init__(self, convnext_model, *args, **kwargs):
        super().__init__()
        self.convnext_model = convnext_model
        model_list = {
            'tiny': convnext_tiny,
            'small': convnext_small,
            'base': convnext_base,
            'large': convnext_large
        }

        backbone = model_list[self.convnext_model](pretrained=True)
        num_filters = backbone.classifier[-1].in_features
        layers = list(backbone.children())[:-1]
        self.backbone = nn.Sequential(*layers)

        num_target_classes = config.NUM_CLASSES
        self.classifier = nn.Linear(num_filters, num_target_classes)

    def forward(self, x):
        x = self.backbone(x).flatten(1)
        x = self.classifier(x)
        return x


class TrashConvnextTiny(TrashConvnext):
    def __init__(self, *args, **kwargs):
        super().__init__('tiny', *args, **kwargs)


class TrashConvnextSmall(TrashConvnext):
    def __init__(self, *args, **kwargs):
        super().__init__('small', *args, **kwargs)


class TrashConvnextBase(TrashConvnext):
    def __init__(self, *args, **kwargs):
        super().__init__('base', *args, **kwargs)


class TrashConvnextLarge(TrashConvnext):
    def __init__(self, *args, **kwargs):
        super().__init__('large', *args, **kwargs)
