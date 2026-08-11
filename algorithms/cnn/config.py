import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "SROIE2019")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train", "img")
TRAIN_BOX_DIR = os.path.join(DATA_ROOT, "train", "box")
TRAIN_ENTITIES_DIR = os.path.join(DATA_ROOT, "train", "entities")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test", "img")
TEST_BOX_DIR = os.path.join(DATA_ROOT, "test", "box")
TEST_ENTITIES_DIR = os.path.join(DATA_ROOT, "test", "entities")

DBNET_DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "SROIE_dbnet")
DBNET_TRAIN_IMG_DIR = os.path.join(DBNET_DATA_ROOT, "train_images")
DBNET_TRAIN_GT_DIR = os.path.join(DBNET_DATA_ROOT, "train_gts")
DBNET_TEST_IMG_DIR = os.path.join(DBNET_DATA_ROOT, "test_images")
DBNET_TEST_GT_DIR = os.path.join(DBNET_DATA_ROOT, "test_gts")
DBNET_TRAIN_LIST = os.path.join(DBNET_DATA_ROOT, "train_list.txt")
DBNET_VAL_LIST = os.path.join(DBNET_DATA_ROOT, "val_list.txt")
DBNET_TEST_LIST = os.path.join(DBNET_DATA_ROOT, "test_list.txt")

VAL_SPLIT_RATIO = 0.1
VAL_SPLIT_SEED = 42

CRNN_DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "SROIE_crnn")
CRNN_TRAIN_CROPS_DIR = os.path.join(CRNN_DATA_ROOT, "train_crops")
CRNN_TEST_CROPS_DIR = os.path.join(CRNN_DATA_ROOT, "test_crops")
CRNN_TRAIN_LABELS = os.path.join(CRNN_DATA_ROOT, "train_labels.txt")
CRNN_VAL_LABELS = os.path.join(CRNN_DATA_ROOT, "val_labels.txt")
CRNN_TEST_LABELS = os.path.join(CRNN_DATA_ROOT, "test_labels.txt")
CRNN_ALPHABET_FILE = os.path.join(CRNN_DATA_ROOT, "alphabet.txt")


MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "cnn")
DBNET_PRETRAINED = os.path.join(MODELS_DIR, "dbnet_pretrained.pth")
DBNET_FINETUNED = os.path.join(MODELS_DIR, "dbnet.pth")
CRNN_WEIGHTS = os.path.join(MODELS_DIR, "crnn.pth")

RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

DBNET_INPUT_SIZE = 640
DBNET_BATCH_SIZE = 4
DBNET_EPOCHS = 50
DBNET_LR = 1e-4
DBNET_BACKBONE = "resnet18"
DBNET_EARLY_STOP_PATIENCE = 10

CRNN_IMG_HEIGHT = 32
CRNN_IMG_MAX_WIDTH = 280
CRNN_HIDDEN_SIZE = 256
CRNN_BATCH_SIZE = 32
CRNN_EPOCHS = 100
CRNN_LR = 1e-3
CRNN_EARLY_STOP_PATIENCE = 10

DEVICE = "cuda"
DET_SCORE_THRESH = 0.5
DET_BOX_THRESH = 0.5