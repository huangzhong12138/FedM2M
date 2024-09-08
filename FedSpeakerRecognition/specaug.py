import numpy as np
import random
import librosa

import scipy.signal as signal
import myconfig


def apply_specaug(features):
    # SpecAugment 训练参数。
    # N_MFCC = 40
    # SEQ_LEN = 100
    # SPECAUG_FREQ_MASK_PROB = 0.3  # 频率遮掩的概率
    # SPECAUG_TIME_MASK_PROB = 0.3  # 时域遮掩的概率
    # SPECAUG_FREQ_MASK_MAX_WIDTH = N_MFCC // 5
    # SPECAUG_TIME_MASK_MAX_WIDTH = SEQ_LEN // 5
    """将 SpecAugment 应用于功能。”"""
    seq_len, n_mfcc = features.shape
    #获取输入特征的形状信息，seq_len：特征序列的长度，n_mfcc：特征向量的维度。
    outputs = features
    mean_feature = np.mean(features)#计算输入特征的平均值

    # 频率掩蔽。
    if random.random() < myconfig.SPECAUG_FREQ_MASK_PROB:
        width = random.randint(1, myconfig.SPECAUG_FREQ_MASK_MAX_WIDTH)
        start = random.randint(0, n_mfcc - width)
        outputs[:, start: start + width] = mean_feature

    # 时域掩蔽。
    if random.random() < myconfig.SPECAUG_TIME_MASK_PROB:
        width = random.randint(1, myconfig.SPECAUG_TIME_MASK_MAX_WIDTH)
        start = random.randint(0, seq_len - width)
        outputs[start: start + width, :] = mean_feature

    return outputs

def denoise(file):
    y = file

    # 计算音频的功率谱
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)) ** 2, ref=np.max)

    # 应用Wiener滤波器进行去噪
    D_denoised = signal.wiener(D, 411)

    # 将去噪后的功率谱转换回音频信号
    y_denoised = librosa.istft(np.exp(librosa.db_to_amplitude(D_denoised)))

    return y_denoised