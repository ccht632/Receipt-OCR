"""
统一配置文件：所有路径和超参数写在这里。
如果你本地的 SROIE 数据集路径和这里不一样，只需要改这个文件，
其他脚本(prepare_*.py / train_*.py / *_detector.py / *_recognizer.py)都不用动。
"""
import os

# ============================================================
# 项目根目录 (自动定位，不用手改)
# ============================================================
# 本文件位于 your_project/algorithms/cnn/config.py
# 所以项目根目录是往上两层
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ============================================================
# 原始 SROIE 数据集路径
# ============================================================
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "SROIE2019")

TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train", "img")
TRAIN_BOX_DIR = os.path.join(DATA_ROOT, "train", "box")
TRAIN_ENTITIES_DIR = os.path.join(DATA_ROOT, "train", "entities")

TEST_IMG_DIR = os.path.join(DATA_ROOT, "test", "img")
TEST_BOX_DIR = os.path.join(DATA_ROOT, "test", "box")
TEST_ENTITIES_DIR = os.path.join(DATA_ROOT, "test", "entities")

# ============================================================
# 转换后的训练数据路径 (由 prepare_*.py 自动生成，不用手建)
# ============================================================
# --- DBNet 检测数据 (ICDAR格式: 图片 + gt.txt，坐标+文字同一行) ---
DBNET_DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "SROIE_dbnet")
DBNET_TRAIN_IMG_DIR = os.path.join(DBNET_DATA_ROOT, "train_images")
DBNET_TRAIN_GT_DIR = os.path.join(DBNET_DATA_ROOT, "train_gts")
DBNET_TEST_IMG_DIR = os.path.join(DBNET_DATA_ROOT, "test_images")
DBNET_TEST_GT_DIR = os.path.join(DBNET_DATA_ROOT, "test_gts")
DBNET_TRAIN_LIST = os.path.join(DBNET_DATA_ROOT, "train_list.txt")
DBNET_VAL_LIST = os.path.join(DBNET_DATA_ROOT, "val_list.txt")
DBNET_TEST_LIST = os.path.join(DBNET_DATA_ROOT, "test_list.txt")

VAL_SPLIT_RATIO = 0.1   # 从train里切10%做validation
VAL_SPLIT_SEED = 42

# --- CRNN 识别数据 (裁剪后的文字行图片 + labels.txt) ---
CRNN_DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "SROIE_crnn")
CRNN_TRAIN_CROPS_DIR = os.path.join(CRNN_DATA_ROOT, "train_crops")
CRNN_TEST_CROPS_DIR = os.path.join(CRNN_DATA_ROOT, "test_crops")
CRNN_TRAIN_LABELS = os.path.join(CRNN_DATA_ROOT, "train_labels.txt")
CRNN_VAL_LABELS = os.path.join(CRNN_DATA_ROOT, "val_labels.txt")
CRNN_TEST_LABELS = os.path.join(CRNN_DATA_ROOT, "test_labels.txt")
CRNN_ALPHABET_FILE = os.path.join(CRNN_DATA_ROOT, "alphabet.txt")

# ============================================================
# 模型权重路径
# ============================================================
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "cnn")
DBNET_PRETRAINED = os.path.join(MODELS_DIR, "dbnet_pretrained.pth")  # 官方预训练权重(手动下载放这)
DBNET_FINETUNED = os.path.join(MODELS_DIR, "dbnet.pth")             # fine-tune后保存到这
CRNN_WEIGHTS = os.path.join(MODELS_DIR, "crnn.pth")                 # 从零训练保存到这

# ============================================================
# 训练日志/输出
# ============================================================
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# ============================================================
# DBNet 超参数
# ============================================================
DBNET_INPUT_SIZE = 640          # 训练时resize的边长(方形)
DBNET_BATCH_SIZE = 4
DBNET_EPOCHS = 50
DBNET_LR = 1e-4                 # fine-tune用小学习率
DBNET_BACKBONE = "resnet18"     # 预训练权重对应的backbone
DBNET_EARLY_STOP_PATIENCE = 10  # val_loss连续10轮不改善就提前停止

# ============================================================
# CRNN 超参数
# ============================================================
CRNN_IMG_HEIGHT = 32            # 输入图片统一高度(经典CRNN设定)
CRNN_IMG_MAX_WIDTH = 280         # 输入图片最大宽度(超过则resize) -- Colab有GPU了，改回280提升精度上限
CRNN_HIDDEN_SIZE = 256           # BiLSTM隐藏层大小 -- Colab有GPU了，改回256提升精度上限
CRNN_BATCH_SIZE = 32
CRNN_EPOCHS = 100
CRNN_LR = 1e-3
CRNN_EARLY_STOP_PATIENCE = 10   # val_CER连续10轮不改善就提前停止

# ============================================================
# 推理相关
# ============================================================
DEVICE = "cuda"  # 训练/推理脚本里会自动检测，没有GPU会fallback成"cpu"
DET_SCORE_THRESH = 0.5   # DBNet 检测框置信度阈值
DET_BOX_THRESH = 0.5     # DBNet 二值化阈值