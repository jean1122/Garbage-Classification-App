NUM_WORKERS = 6
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
NUM_RUNS = 10
LOG_EVERY_N_STEPS = 10
REMOVE_PREFIX = True

LOG_TRAIN_STEP_ACCURACY = True
LOG_TRAIN_EPOCH_ACCURACY = True
LOG_VAL_STEP_ACCURACY = True
LOG_VAL_EPOCH_ACCURACY = True

CHECKPOINT_DIR = 'checkpoints'
ROOT_DIR = '/mnt/d/Datasets/trashnet/resized'

PROJECT_NAME = 'trash-ai'

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {
        "name": "val_loss",
        "goal": "minimize"
    },
    "parameters": {
        "model": {
            "values": ["efficientnet_b0"]
        },
        # "model": {
        #     "values": ["resnet101", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2"]
        # },
        "batch_size": {
            "values": [16, 32, 64, 128]
        },
        "lr": {
            # log uniform distribution between exp(min) and exp(max)
            "distribution": "log_uniform",
            "min": -9.21,   # exp(-9.21) = 1e-4
            "max": -4.61    # exp(-4.61) = 1e-2
        }
    }
}
