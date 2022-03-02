import torch
import torch.nn as nn
import config
from models.base_model import TrashBaseClass


class TrashBaseline(TrashBaseClass):
    def __init__(self, *args, **kwargs):
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.IMAGE_SIZE * config.IMAGE_SIZE *
                      config.INPUT_CHANNELS, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, config.NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)


def trash_baseline(*args, **kwargs):
    return TrashBaseline(*args, **kwargs)
