# This file has the configurations of the experiments.
import os
import torch
import multiprocessing

# Paths of downloaded LibriSpeech datasets.
# TRAIN_DATA_DIR = os.path.join(
#     os.path.expanduser("~"),
#     "SpeakerRecognitionFromScratch-main/SpeakerRecognitionFromScratch-main/data/train-clean-100/LibriSpeech/train-clean-100")
# TEST_DATA_DIR = os.path.join(
#     os.path.expanduser("~"),
#     "SpeakerRecognitionFromScratch-main/SpeakerRecognitionFromScratch-main/data/test-clean/LibriSpeech/test-clean")
TRAIN_DATA_DIR ="LibriSpeech/train-clean-100"
TEST_DATA_DIR = "data/test-clean/LibriSpeech/test-clean"


# Paths of CSV files where the first column is speaker, and the second column is
# utterance file.
# These will allow you to train/evaluate using other datasets than LibriSpeech.
# If given, TRAIN_DATA_DIR and/or TEST_DATA_DIR will be ignored.
TRAIN_DATA_CSV = ""
TEST_DATA_CSV = ""

# Path of save model.
# SAVED_MODEL_PATH = os.path.join(
#     os.path.expanduser("~"),
#     "SpeakerRecognitionFromScratch-main/SpeakerRecognitionFromScratch-main/saved_model/me/saved_model.pt")
SAVED_MODEL_PATH ="saved_model/me/saved_model.noise.10.5.pt"


# librosa.feature.mfcc 的 MFCC 数量。
N_MFCC = 40


# LSTM 层的隐藏大小。
LSTM_HIDDEN_SIZE = 64

# LSTM 层.
LSTM_NUM_LAYERS = 3

# 是否使用双向 LSTM。
BI_LSTM = True

# 如果为假，则使用 LSTM 推理的最后一帧作为聚合输出；
# 如果为真，则使用 LSTM 推理的平均帧作为聚合输出。
FRAME_AGGREGATION_MEAN = True

# 如果为真，我们将使用 transformer 而不是 LSTM。
USE_TRANSFORMER = False

# Dimension of transformer layers.
TRANSFORMER_DIM = 32

# Number of encoder layers for transformer
TRANSFORMER_ENCODER_LAYERS = 2

# Number of heads in transformer layers.
TRANSFORMER_HEADS = 8

# LSTM 滑动窗口的序列长度。
SEQ_LEN = 100  # 3.2 seconds

# Alpha for the triplet loss.三元组损失的阿尔法。
TRIPLET_ALPHA = 0.1

# How many triplets do we train in a single batch.
# 我们在一个批次中训练了多少个三元组。
BATCH_SIZE = 8

# 学习率
LEARNING_RATE = 0.001

# 每隔这些步骤将模型保存到磁盘。
SAVE_MODEL_FREQUENCY = 10000

# 要训练的步数。
TRAINING_STEPS = 10000

# Whether we are going to train with SpecAugment.
# 我们是否要使用 SpecAugment 进行训练。
SPECAUG_TRAINING = True

# SpecAugment 训练参数。
SPECAUG_FREQ_MASK_PROB = 0.3#频率遮掩的概率
SPECAUG_TIME_MASK_PROB = 0.3#时域遮掩的概率
SPECAUG_FREQ_MASK_MAX_WIDTH = N_MFCC // 5
SPECAUG_TIME_MASK_MAX_WIDTH = SEQ_LEN // 5

# 是否使用全序列推理或滑动窗口推理。
USE_FULL_SEQUENCE_INFERENCE = False

# 用于滑动窗口推理的滑动窗口步骤。
SLIDING_WINDOW_STEP = 50  # 1.6 seconds

# 为计算等错误率 (EER) 而评估的三元组数。
# 正面试验的数量和负面试验的数量都将是
# 等于这个数字。
NUM_EVAL_TRIPLETS = 10000

# 用于计算等错误率 (EER) 的阈值扫描步骤。
EVAL_THRESHOLD_STEP = 0.001

# 多处理的进程数。
NUM_PROCESSES = min(multiprocessing.cpu_count(), BATCH_SIZE)

# Wehther to use GPU or CPU.
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
