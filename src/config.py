import torch

DATA_DIR = "data"
IMG_HEIGHT = IMG_WIDTH = 200
BATCH_SIZE = 50
EPOCHS = 10
NUM_WORKERS = 4
PIN_MEMORY = True
LABEL_MAP = {"cat": 0, "dog": 1}
FOLDS = 5
COSINE_ANNEALING_T0 = 10
LR = 1e-3
MODEL_DIR = "models"
INFERENCE_DIR = "inferences"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
