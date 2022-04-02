import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import BackboneFinetuning
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from dataset import *
import models

torch.manual_seed(42)

plt.rcParams['figure.figsize'] = (config.FIGURE_SIZE, config.FIGURE_SIZE)

model_dict = models.model_dict

model_name = 'resnet152'
model_cls = model_dict[model_name]
model = model_cls(lr=0.01)

dm = TrashNetDataModuleWithResized(transfer_learning=model.transfer_learning, batch_size=64)
backbone_fine_tuning = BackboneFinetuning(config.START_BACKBONE_TUNING_EPOCH, lambda epoch: 1.25)
trainer = pl.Trainer(max_epochs=2, gpus=config.NUM_GPUS, profiler="simple", callbacks=[backbone_fine_tuning],enable_checkpointing=False)
trainer.fit(model, dm)
