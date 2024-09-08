import numpy as np
import torch
import torchaudio


def add_guassian_snr(audio_file_path, output_path, target_snr_db):
    audio, sample_rate = torchaudio.load(audio_file_path)
    transform_model = torchaudio.transforms.Resample(sample_rate, 16000)
    audio = transform_model(audio)
    # if the audio is not mono channel
    if audio.shape[0] == 2:
        audio = torch.mean(audio, dim=0).unsqueeze(dim=0)
    audio_energy_watts = np.mean(audio.detach().cpu().numpy() ** 2)  # 计算音频能量（瓦特的平方）
    audio_energy_db = 10 * np.log10(audio_energy_watts)  # 将音频能量转换为分贝（dB）

    noise_energy_db = audio_energy_db - target_snr_db  # 计算目标信噪比对应的噪声能量（分贝）
    noise_energy_watts = 10 ** (noise_energy_db / 10)  # 将噪声能量转换为瓦特平方。

    np.random.seed(8)  # 设置随机数
    noise = np.random.normal(0, np.sqrt(noise_energy_watts), audio.shape[1]) # 生成高斯噪声
    noise_audio = audio + noise
    noise_audio = noise_audio.type(torch.float32)
    torchaudio.save(output_path, noise_audio, sample_rate)

