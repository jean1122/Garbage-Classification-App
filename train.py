import config
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from dataset import TrashNetDataModule
import models

model = models.TrashEfficientnetB0(lr=0.001)
print(model.model_size)
# dm = TrashNetDataModule(
#     transfer_learning=model.transfer_learning, batch_size=config.BATCH_SIZE)


# stop_checkpoint = EarlyStopping('val_loss')
# trainer = pl.Trainer(max_epochs=config.EPOCHS, gpus=config.NUM_GPUS,
#                      log_every_n_steps=config.LOG_EVERY_N_STEPS, callbacks=[stop_checkpoint], profiler="simple")

# trainer.fit(model, dm)
