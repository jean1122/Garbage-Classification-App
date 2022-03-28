NUM_WORKERS = 0
EPOCHS = 200
BATCH_SIZE = 64
VAL_SPLIT = 0.1
TEST_SPLIT = 0.2
PIN_MEMORY = True
IMAGE_SIZE = 64
INPUT_CHANNELS = 3
NUM_CLASSES = 6
B1 = 0.9
B2 = 0.99
NUM_GPUS = 1
FIGURE_SIZE = 12
NUM_RUNS = 20
LOG_EVERY_N_STEPS = 10
REMOVE_PREFIX = True

LOG_TRAIN_STEP_ACCURACY = True
LOG_TRAIN_EPOCH_ACCURACY = True
LOG_VAL_STEP_ACCURACY = True
LOG_VAL_EPOCH_ACCURACY = True

CHECKPOINT_DIR = 'checkpoints'
ROOT_DIR = '/mnt/d/Datasets/trashnet/resized'

PROJECT_NAME = 'trash-ai'

# Dictionary that stores the W&B sweep ids. After training a model,
# place the model name and sweep id in this dictionary, so that the same
# model can be run with the same sweep
SWEEP_ID = {
    'baseline': 'bw2k0kvv',
    'efficientnet_b7': 'gnijdvsm',
    'efficientnet_b1' : 'dz3xe8aj',
    'efficientnet_b0': 'rwb5lurg',
    'convnext_tiny': 'cdhofwn6',
    'convnext_base': 'jl4l1frd',
    'convnext_large': '9rrdpjzq',
    'vit_b16': '86a3oe18',
}

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {
        "name": "val_loss",
        "goal": "minimize"
    },
    "parameters": {
        "model": {
            "value": "convnext_base"
        },
        "batch_size": {
            "values": [16, 32, 64]
        },
        "lr": {
            # log uniform distribution between exp(min) and exp(max)
            "distribution": "log_uniform",
            "min": -9.21,   # exp(-9.21) = 1e-4
            "max": -4.61    # exp(-4.61) = 1e-2
        }
    }
}
