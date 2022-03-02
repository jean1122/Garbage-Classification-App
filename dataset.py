import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import config
import pytorch_lightning as pl

class DataAugmentation(nn.Module):
    def __init__(self, apply_color_jitter = False, apply_random_augment = True, *args, **kwarrgs):
        super().__init__()
        self._apply_color_jitter = apply_color_jitter
        self._apply_random_augment = apply_random_augment
        
        self.rand_augment = transforms.RandAugment(*args, **kwarrgs)
        self.jitter = transforms.ColorJitter(0.5, 0.5, 0.5, 0.5)
    
    @torch.no_grad()
    def forward(self, x):
        if self._apply_color_jitter:
            x = self.jitter(x)
        if self._apply_random_augment:
            x = self.rand_augment(x)
        return x

class TrashNetDataModule(pl.LightningDataModule):
    def __init__(self, transfer_learning=False, augment=True, data_dir=config.ROOT_DIR, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.augment = augment
        self.image_size = 224 if transfer_learning else config.IMAGE_SIZE
        # mean and standard deviations computed using seed = 42
        self.mean, self.std = (
            0.5389, 0.5123, 0.4846), (0.2201, 0.2178, 0.2323)

        self.augmentation = DataAugmentation()

    # Get indices of the train, validation, and test dataset split equally according to class distribution

    def get_indices(self, dataset):
        targets = np.asarray(dataset.targets)
        train_data_idx, test_idx = train_test_split(
            np.arange(len(targets)), test_size=config.TEST_SPLIT, stratify=targets)
        train_idx, val_idx = train_test_split(np.arange(len(
            train_data_idx)), test_size=config.VAL_SPLIT, stratify=targets[train_data_idx])
        train_idx, val_idx = train_data_idx[train_idx], train_data_idx[val_idx]
        return train_idx, val_idx, test_idx

    # Get samplers from indices
    def get_samplers(self, train_idx, val_idx, test_idx):
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        return train_sampler, val_sampler, test_sampler

    def setup(self, stage=None):
        dataset = datasets.ImageFolder(config.ROOT_DIR, transform=transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ]))

        train_idx, val_idx, test_idx = self.get_indices(dataset)

        # Only calculate the mean and std of train and val dataset. Test idx is hidden.
        # self.mean, self.std = self.get_distribution(
        #     dataset, np.concatenate([train_idx, val_idx]))
        self.train_sampler, self.val_sampler, self.test_sampler = self.get_samplers(
            train_idx, val_idx, test_idx)

    def train_dataloader(self):
        transform = [self.augmentation] if self.augment else []
        transform = transforms.Compose(transform + [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        dataset = datasets.ImageFolder(config.ROOT_DIR, transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size, sampler=self.train_sampler, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        dataset = datasets.ImageFolder(config.ROOT_DIR, transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size,
                          sampler=self.val_sampler, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        dataset = datasets.ImageFolder(config.ROOT_DIR, transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size,
                          sampler=self.test_sampler, num_workers=self.num_workers, pin_memory=self.pin_memory)

    