import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import BackboneFinetuning
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from dataset import *
import wandb
import models

torch.manual_seed(42)

wandb.login()

batch_size = -1 # Set to desired batch_size
learning_rate = -1 # Set to desired learning_rate
model_name = '' # Set to model name
assert model_name in models.model_dict

wandb.init()  # required to have access to `wandb.config`
wandb_logger = WandbLogger()

model_cls = models.model_dict[model_name]
model = model_cls(lr=learning_rate)

dm = TrashNetDataModuleWithResized(transfer_learning=model_cls.transfer_learning, batch_size=batch_size)

stop_checkpoint = EarlyStopping('val_loss', patience=8)
backbone_fine_tuning = BackboneFinetuning(config.START_BACKBONE_TUNING_EPOCH, lambda epoch: 1.25)
trainer = pl.Trainer(max_epochs=config.EPOCHS, logger=wandb_logger, gpus=config.NUM_GPUS,
                     log_every_n_steps=config.LOG_EVERY_N_STEPS, callbacks=[stop_checkpoint, backbone_fine_tuning],
                     profiler="simple", enable_checkpointing=False)
trainer.fit(model, dm)


