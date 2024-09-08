import soundfile as sf
import librosa
import numpy as np
import specaug
from sklearn.preprocessing import StandardScaler

def extract_features(audio_file):
    N_MFCC = 40


    """Extract MFCC features from an audio file, shape=(TIME, MFCC).
    从音频文件中提取 MFCC 特征，shape=(TIME, MFCC)。"""
    waveform, sample_rate = sf.read(audio_file)
    # 设置一个降噪手段
    # waveform = specaug.denoise(waveform)

    # Convert to mono-channel.
    # 转换为单声道librosa.to_mono()是一个函数，用于将立体声音频数据转换为单声道。
    if len(waveform.shape) == 2:
        waveform = librosa.to_mono(waveform.transpose())

    # Convert to 16kHz.转换为 16kHz
    if sample_rate != 16000:
        waveform = librosa.resample(waveform, sample_rate, 16000)


    # 使用Librosa库计算音频文件的梅尔频率倒谱系数（MFCC）特征。
    features = librosa.feature.mfcc(
        y=waveform, sr=sample_rate, n_mfcc=N_MFCC, n_fft=1024)

    # 归一化 MFCC 特征
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features.transpose())
    return features_normalized




def clip_features(audio,target_duration):
    # 随机剪裁音频以满足目标持续时间
    if len(audio) <= target_duration * 1000:
        return audio
    else:
        max_start = len(audio) - target_duration * 1000
        start = np.random.randint(0, max_start)
        end = start + target_duration * 1000
        return audio[start:end]