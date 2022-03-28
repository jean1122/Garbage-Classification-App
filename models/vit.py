import torch
import torch.nn as nn
import config
from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32

from models.base_model import TrashBaseClass


class TrashVit(TrashBaseClass):
    transfer_learning = True

    def __init__(self, vit_model, n_heads, *args, **kwargs):
        super().__init__()
        self.vit_model = vit_model
        self.n_heads = n_heads
        
        model_list = {
            'vit_b': {
                16: vit_b_16,
                32: vit_b_32
            },
            'vit_l': {
                16: vit_l_16,
                32: vit_l_32
            }
        }

        self.backbone = model_list[self.vit_model][self.n_heads](pretrained=True)

        if self.backbone.representation_size is None:
            num_filters = self.backbone.hidden_dim
        else:
            num_filters = self.backbone.representation_size

        num_target_classes = config.NUM_CLASSES
        self.backbone.heads[-1] = nn.Identity(num_filters, num_filters)
        self.classifier = nn.Linear(num_filters, num_target_classes)

    def get_features(self, x):
        self.backbone.eval()
        with torch.no_grad():
            x = self.backbone(x).flatten(1)
        return x

    def forward(self, x):
        representations = self.get_features(x)
        x = self.classifier(representations)
        return x

class TrashVitB16(TrashVit):
    def __init__(self, *args, **kwargs):
        super().__init__('vit_b', 16, *args, **kwargs)


class TrashVitB32(TrashVit):
    def __init__(self, *args, **kwargs):
        super().__init__('vit_b', 32, *args, **kwargs)


class TrashVitL16(TrashVit):
    def __init__(self, *args, **kwargs):
        super().__init__('vit_l', 16, *args, **kwargs)


class TrashVitL32(TrashVit):
    def __init__(self, *args, **kwargs):
        super().__init__('vit_l', 32, *args, **kwargs)