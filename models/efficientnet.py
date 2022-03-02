import torch
import torch.nn as nn
import config
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7

from models.base_model import TrashBaseClass


class TrashEfficientnet(TrashBaseClass):
    transfer_learning = True

    def __init__(self, efficientnet_model, *args, **kwargs):
        super().__init__()
        self.efficientnet_model = efficientnet_model

        model_list = [efficientnet_b0, efficientnet_b1,
                      efficientnet_b2, efficientnet_b3,
                      efficientnet_b4, efficientnet_b5,
                      efficientnet_b6, efficientnet_b7]
        # Can test different efficientnet settings
        backbone = model_list[self.efficientnet_model](pretrained=True)
        p_dropout = backbone.classifier[0].p
        num_filters = backbone.classifier[-1].in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        num_target_classes = config.NUM_CLASSES
        self.classifier = nn.Sequential(
            nn.Dropout(p=p_dropout, inplace=True),  # Dropout value given by b4
            nn.Linear(num_filters, num_target_classes)
        )

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x


class TrashEfficientnetB0(TrashEfficientnet):
    def __init__(self, *args, **kwargs):
        super().__init__(0, *args, **kwargs)


class TrashEfficientnetB1(TrashEfficientnet):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)


class TrashEfficientnetB2(TrashEfficientnet):
    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)


class TrashEfficientnetB3(TrashEfficientnet):
    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)


class TrashEfficientnetB4(TrashEfficientnet):
    def __init__(self, *args, **kwargs):
        super().__init__(4, *args, **kwargs)


class TrashEfficientnetB5(TrashEfficientnet):
    def __init__(self, *args, **kwargs):
        super().__init__(5, *args, **kwargs)


class TrashEfficientnetB6(TrashEfficientnet):
    def __init__(self, *args, **kwargs):
        super().__init__(6, *args, **kwargs)


class TrashEfficientnetB7(TrashEfficientnet):
    def __init__(self, *args, **kwargs):
        super().__init__(7, *args, **kwargs)
