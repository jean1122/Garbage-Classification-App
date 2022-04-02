import torch
import torch.nn.functional as F
import config
import pytorch_lightning as pl
import torchmetrics
import re


class TrashBaseClass(pl.LightningModule):
    transfer_learning = False
    model_dict = dict()

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

    def __init_subclass__(cls):
        cls.model_dict[cls.get_name(config.REMOVE_PREFIX)] = cls

    def save_metrics(self, metric, mode, pred, labels, loss):
        metric(pred, labels)
        metrics = {f'{mode}_accuracy': metric, f'{mode}_loss': loss}
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr, betas=(config.B1, config.B2))
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            "monitor": 'val_loss'
        }

    @staticmethod
    def snake_case(string, remove_prefix=True):
        string = re.sub(r'(?<!^)(?=[A-Z])', '_', string).lower()
        if remove_prefix:
            string = string.replace('trash_', '')
        return string

    @classmethod
    def get_name(cls, remove_prefix=True):
        return cls.snake_case(cls.__name__, remove_prefix)

    @classmethod
    def get_model_from_name(cls, name):
        if name in cls.model_dict:
            return cls.model_dict[name]
        return None

    def forward_step(self, batch):
        imgs, labels = batch
        pred = self(imgs).squeeze()
        loss = F.cross_entropy(pred, labels)
        return pred, loss, labels

    def training_step(self, batch, batch_idx):
        pred, loss, labels = self.forward_step(batch)
        self.save_metrics(self.train_accuracy, 'train', pred, labels, loss)
        return loss

    def validation_step(self, batch, batch_idx):
        pred, loss, labels = self.forward_step(batch)
        metrics = self.save_metrics(self.val_accuracy, 'val', pred, labels, loss)
        return metrics

    def test_step(self, batch, batch_idx):
        pred, loss, labels = self.forward_step(batch)
        metrics = self.save_metrics(self.test_accuracy, 'test', pred, labels, loss)
        return metrics
