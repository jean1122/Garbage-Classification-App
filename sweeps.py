import torch
import matplotlib.pyplot as plt
import config
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from dataset import TrashNetDataModule
import wandb
import models

torch.manual_seed(42)
plt.rcParams['figure.figsize'] = (config.FIGURE_SIZE, config.FIGURE_SIZE)

model_dict = models.model_dict

wandb.login()


def check_models_exist(sweep_config):
    model_list = sweep_config['parameters']['model']['values']
    assert all([model_name in model_dict for model_name in model_list])


def sweep_iteration():
    wandb.init()  # required to have access to `wandb.config`
    wandb_logger = WandbLogger()

    model_cls = model_dict[wandb.config.model]
    model = model_cls(lr=wandb.config.lr)

    dm = TrashNetDataModule(
        transfer_learning=model_cls.transfer_learning, batch_size=wandb.config.batch_size)

    stop_checkpoint = EarlyStopping('val_loss')
    wandb_logger.watch(model)
    trainer = pl.Trainer(max_epochs=config.EPOCHS, logger=wandb_logger, gpus=config.NUM_GPUS,
                         log_every_n_steps=config.LOG_EVERY_N_STEPS, callbacks=[stop_checkpoint], profiler="simple")

    trainer.fit(model, dm)


check_models_exist(config.SWEEP_CONFIG)
sweep_id = wandb.sweep(config.SWEEP_CONFIG, project=config.PROJECT_NAME)
wandb.agent(sweep_id, function=sweep_iteration, count=config.NUM_RUNS)
