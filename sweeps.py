import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import BackboneFinetuning
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from dataset import *
import wandb
import models

torch.manual_seed(42)
plt.rcParams['figure.figsize'] = (config.FIGURE_SIZE, config.FIGURE_SIZE)

model_dict = models.model_dict

wandb.login()


def check_models_exist(sweep_config):
    name = sweep_config['parameters']['model']['value']
    assert name in model_dict


def sweep_iteration():
    wandb.init()  # required to have access to `wandb.config`
    wandb_logger = WandbLogger()

    model_cls = model_dict[wandb.config.model]
    model = model_cls(lr=wandb.config.lr)

    dm = TrashNetDataModuleWithResized(
        transfer_learning=model_cls.transfer_learning, batch_size=wandb.config.batch_size)

    stop_checkpoint = EarlyStopping('val_loss', patience=8)
    backbone_fine_tuning = BackboneFinetuning(config.START_BACKBONE_TUNING_EPOCH, lambda epoch: 1.25)
    trainer = pl.Trainer(max_epochs=config.EPOCHS, logger=wandb_logger, gpus=config.NUM_GPUS,
                         log_every_n_steps=config.LOG_EVERY_N_STEPS, callbacks=[stop_checkpoint, backbone_fine_tuning],
                         profiler="simple", enable_checkpointing=False)
    wandb.watch(model)
    trainer.fit(model, dm)


check_models_exist(config.SWEEP_CONFIG)

model_name = config.SWEEP_CONFIG['parameters']['model']['value']

sweepID = config.SWEEP_ID if config.PROJECT_NAME == 'trash-ai' else config.SWEEP_ID_BACKBONE

if model_name in sweepID:
    sweep_id = f'{config.PROJECT_NAME}/{config.SWEEP_ID[model_name]}'
else:
    sweep_id = wandb.sweep(config.SWEEP_CONFIG, project=config.PROJECT_NAME)
wandb.agent(sweep_id, function=sweep_iteration, count=config.NUM_RUNS)
