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
NUM_RUNS = 15
LOG_EVERY_N_STEPS = 10
REMOVE_PREFIX = True
START_BACKBONE_TUNING_EPOCH = 25

CHECKPOINT_DIR = 'checkpoints'
ROOT_DIR = '/mnt/d/Datasets/trashnet/resized'

PROJECT_NAME = 'trash-ai-backbone-fine-tuning'

# Dictionary that stores the W&B sweep ids. After training a model,
# place the model name and sweep id in this dictionary, so that the same
# model can be run with the same sweep
SWEEP_ID = {
    'baseline': 'bw2k0kvv',
    'efficientnet_b7': 'gnijdvsm',
    'efficientnet_b1' : 'dz3xe8aj',
    'efficientnet_b0': 'rwb5lurg',
    'resnet152': '7urzgc8j',
    'convnext_tiny': 'cdhofwn6',
    'convnext_small': 'c22earkk',
    'convnext_base': 'jl4l1frd',
    'convnext_large': 'luemjuu7',
    'vit_b16': 'eljnsu2y',
    'vit_b32': 'otls3zlv',
    'vit_l16': 'ih9w5chx',
    'vit_l32': 'memzurmx'
}

SWEEP_ID_BACKBONE = {

}

model_name = 'resnet18'

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {
        "name": "val_loss",
        "goal": "minimize"
    },
    "parameters": {
        "model": {
            "value": model_name
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
