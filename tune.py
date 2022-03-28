import matplotlib.pyplot as plt
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from dataset import *
import models
from models import TrashBaseline

torch.manual_seed(42)

plt.rcParams['figure.figsize'] = (config.FIGURE_SIZE, config.FIGURE_SIZE)

model_dict = models.model_dict

model_name = 'vit_l32'
model_cls = model_dict[model_name]
model = model_cls(lr=0.01)

dm = TrashNetDataModuleWithResized(transfer_learning=model.transfer_learning, batch_size=64)

trainer = pl.Trainer(max_epochs=2, gpus=config.NUM_GPUS, profiler="simple")
try:
    trainer.fit(model, dm)
except RuntimeError as e:
    print(e)
