import glob
import os
import random
import numpy as np
import torch
import specaug
from datautil.audio_features import extract_features
from torch.utils.data import Dataset
from datautil.datasplit import getdataloader


# class AudioDataset(Dataset):
#     def __init__(self, data_dir, label_mapping, transform=None, duration=150):
#         self.transform = transform
#         self.data_dir = data_dir
#         self.label_mapping = label_mapping
#         self.target_duration = duration
#         # 加载声纹数据
#         # data, target = self.load_data()
#         self.data, self.targets = self.load_data()
#         self.data = self.data
#         self.targets = np.squeeze(self.targets)
#
#         # 自动打乱数据顺序
#         self.shuffle_data()
#
#     def load_data(self):
#         # 接收列表，[0]为地址、[1]为说话人标签
#         data = []
#         labels = []
#         for i in range(len(self.data_dir)):
#             root, speaker = self.data_dir[i]
#
#             mfcc = extract_features(root)
#             # 获取音频的总长度
#             total_length = mfcc.shape[0]
#             desired_length = self.target_duration  # 你期望的音频特征长度
#             # 裁剪静音
#
#             step = 50  # 滑动窗口的步长
#
#             # 开始剪裁音频
#             desired_length = self.target_duration  # 你期望的音频特征长度
#             if mfcc.shape[0] < desired_length:
#                 # 填充音频特征
#                 padding = np.zeros((desired_length - mfcc.shape[0], mfcc.shape[1]))
#                 mfcc = np.vstack((mfcc, padding))
#             elif mfcc.shape[0] > desired_length:
#                 # 随机选择起点
#                 start_point = random.randint(0, total_length - desired_length)
#                 # 从随机起点开始抽取所需长度的音频特征
#                 mfcc = mfcc[start_point:start_point + desired_length, :]
#                 # # 剪裁音频特征
#                 # mfcc = mfcc[:desired_length, :]
#                 # 以上是随机选取片段
#
#             # 标签映射
#             # print("Speaker:", speaker)
#             # print("str(speaker):", str(speaker))
#             if str(speaker) in self.label_mapping:
#                 # speaker_to_labelmap = self.label_mapping.get(speaker,-1)
#                 speaker_to_labelmap = self.label_mapping[str(speaker)]
#                 # if speaker_to_labelmap !=-1:
#                 data.append(mfcc)
#                 labels.append(speaker_to_labelmap)
#             else:
#                 continue
#
#         return data, labels
#
#     def __len__(self):
#         return len(self.data)
#
#     def shuffle_data(self):
#         # 随机打乱数据顺序
#         indices = list(range(len(self.data)))
#         random.shuffle(indices)
#         self.data = [self.data[i] for i in indices]
#         self.targets = [self.targets[i] for i in indices]
#
#     def __getitem__(self, idx):
#         sample = self.data[idx]
#         label = self.targets[idx]
#         if self.transform:
#             sample = self.transform(sample)
#
#         return sample, label

class AudioDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy') and '_label' not in f]

        self.data, self.targets = self.load_data()
        self.data = self.data
        self.targets = np.squeeze(self.targets)

    def __len__(self):
        return len(self.file_list)

    def load_data(self):
        data_list = []
        labels_list = []
        for i in range(len(self.file_list)):
            data, labels = self.__getitem__(i)
            data_list.append(data)
            labels_list.append(labels)
        return data_list, labels_list

    def __getitem__(self, idx):
        feature_file = os.path.join(self.data_dir, self.file_list[idx])
        features = np.load(feature_file)

        label_file = feature_file.replace('.npy', '_label.npy')
        targets = np.load(label_file)

        return features, targets


# 将音频预处理方法独立
def audio_feature_method(data_dir, label_mapping, preprocessed_dir, target_duration=50):
    # 接收列表，[0]为地址、[1]为说话人标签
    # data = []
    # labels = []

    for i in range(len(data_dir)):
        root, speaker = data_dir[i]
        mfcc = extract_features(root)
        # 获取音频的总长度
        total_length = mfcc.shape[0]

        # 开始剪裁音频
        desired_length = target_duration  # 你期望的音频特征长度
        if mfcc.shape[0] < desired_length:
            # 填充音频特征
            padding = np.zeros((desired_length - mfcc.shape[0], mfcc.shape[1]))
            mfcc = np.vstack((mfcc, padding))
        elif mfcc.shape[0] > desired_length:
            # 随机选择起点
            start_point = random.randint(0, total_length - desired_length)
            # 从随机起点开始抽取所需长度的音频特征
            mfcc = mfcc[start_point:start_point + desired_length, :]

        if str(speaker) in label_mapping:
            # speaker_to_labelmap = self.label_mapping.get(speaker,-1)
            speaker_to_labelmap = label_mapping[str(speaker)]
            # if speaker_to_labelmap !=-1:

            # data.append(mfcc)
            # labels.append(speaker_to_labelmap)

            feature_file = os.path.join(preprocessed_dir, f'{speaker}_{i}.npy')  # preprocessed_dir是保存文件位置
            np.save(feature_file, mfcc)
            label_file = feature_file.replace('.npy', '_label.npy')
            np.save(label_file, speaker_to_labelmap)
        else:
            continue


def zhvoice_data(file_address, args):
    print("zhvoice_data_to_text方法成功被调用")
    flac_files = glob.glob(os.path.join(file_address, "*", "*", "*.mp3"))
    print(flac_files)
    spk_to_utts = dict()
    speakers = []  # 唯一说话人标记的列表，为之后分组客户端用
    for flac_file in flac_files:
        split_name = flac_file.split('\\')
        spk = split_name[2]
        if spk not in spk_to_utts:
            spk_to_utts[spk] = [flac_file.replace('\\', '/')]
            speakers.append(spk)
        else:
            spk_to_utts[spk].append(flac_file.replace('\\', '/'))

    args.num_classes = len(speakers)
    # print(spk_to_utts)
    # print(speakers)
    # print(len(speakers))
    return spk_to_utts, speakers

def preprocessed_dir_method(preprocessed_dir, args):
    # 定义目录路径
    directory = args.audio_file_address
    # 获取路径中的最后一个文件名
    last_file = os.path.basename(directory)
    # 输出最后一个文件名
    print("最后一个文件名是:", last_file)
    if last_file =='zhthchs30':
        preprocessed_dir = 'preprocessed_data_zhvoice_zhthchs30'
    elif last_file =='zhstcmds_shengxia':
        preprocessed_dir = 'preprocessed_data_zhvoice_zhstcmds_shengxia'
    # elif last_file =='zhstcmds_shengxia':
    return preprocessed_dir

def zhvoice_data_method(args):
    spk_to_utts, speakers = zhvoice_data(args.audio_file_address, args)
    class_to_id = classses(speakers)
    address_speakers = address_speakers_to_list(spk_to_utts, speakers)
    # data = AudioDataset(data_dir=address_speakers, label_mapping=class_to_id)

    # 新加的
    preprocessed_dir = 'preprocessed_data_zhvoice_zhthchs30'
    preprocessed_dir = preprocessed_dir_method(preprocessed_dir, args)
    if not os.path.exists(preprocessed_dir):
        os.makedirs(preprocessed_dir)

    if os.path.exists(preprocessed_dir) and os.listdir(preprocessed_dir):
        panduan = True
    else:
        panduan = False
    if not panduan:
        audio_feature_method(address_speakers, class_to_id, preprocessed_dir)
    data = AudioDataset(preprocessed_dir)
    trd, vad, ted = getlabeldataloader(args, data)
    print("-------------------------------")
    print("传入后的args.batch:{}".format(args.batch))

    print("args.num_classes的值是：{}".format(args.num_classes))
    return trd, vad, ted


def timit_data_method(args):
    spk_to_utts, speakers = timit_data(args.audio_file_address, args)
    class_to_id = classses(speakers)
    address_speakers = address_speakers_to_list(spk_to_utts, speakers)
    data = AudioDataset(data_dir=address_speakers, label_mapping=class_to_id)

    trd, vad, ted = getlabeldataloader(args, data)
    print("-------------------------------")
    print("传入后的args.batch:{}".format(args.batch))

    print("args.num_classes的值是：{}".format(args.num_classes))
    return trd, vad, ted


def timit_data(file_address, args):
    print("timit_data_to_text方法成功被调用")
    flac_files = glob.glob(os.path.join(file_address, "*", "*", "*.WAV"))
    # print(flac_files)
    spk_to_utts = dict()
    speakers = []  # 唯一说话人标记的列表，为之后分组客户端用
    for flac_file in flac_files:
        split_name = flac_file.split('\\')
        spk = split_name[2]
        if spk not in spk_to_utts:
            spk_to_utts[spk] = [flac_file.replace('\\', '/')]
            speakers.append(spk)
        else:
            spk_to_utts[spk].append(flac_file.replace('\\', '/'))

    args.num_classes = len(speakers)
    # print(spk_to_utts)
    # print(speakers)
    # print(len(speakers))
    return spk_to_utts, speakers


def voxceleb1_data(file_address, args):
    print("voxceleb1_data_to_text方法成功被调用")
    flac_files = glob.glob(os.path.join(file_address, "*", "*", "*.wav"))
    spk_to_utts = dict()
    speakers = []  # 唯一说话人标记的列表，为之后分组客户端用
    for flac_file in flac_files:
        split_name = flac_file.split('\\')
        # print(split_name[1])
        spk = split_name[1]
        if spk not in spk_to_utts:
            spk_to_utts[spk] = [flac_file.replace('\\', '/')]
            speakers.append(spk)
        else:
            spk_to_utts[spk].append(flac_file.replace('\\', '/'))
    args.num_classes = len(speakers)
    # print(spk_to_utts)
    # print(speakers)
    # print(len(speakers))
    return spk_to_utts, speakers


def voxceleb2_data(file_address, args):
    print("voxceleb2_data_to_text方法成功被调用")
    flac_files = glob.glob(os.path.join(file_address, "*", "*", "*.wav"))
    spk_to_utts = dict()
    speakers = []  # 唯一说话人标记的列表，为之后分组客户端用
    for flac_file in flac_files:
        split_name = flac_file.split('\\')
        # print(split_name[1])
        spk = split_name[1]
        if spk not in spk_to_utts:
            spk_to_utts[spk] = [flac_file.replace('\\', '/')]
            speakers.append(spk)
        else:
            spk_to_utts[spk].append(flac_file.replace('\\', '/'))
    args.num_classes = len(speakers)
    # print(spk_to_utts)
    # print(speakers)
    # print(len(speakers))
    return spk_to_utts, speakers


def librispeech_data_method(args):
    spk_to_utts, speakers = librispeech_data(args.audio_file_address, args)
    class_to_id = classses(speakers)
    address_speakers = address_speakers_to_list(spk_to_utts, speakers)
    # data = AudioDataset(data_dir=address_speakers, label_mapping=class_to_id)

    # 新加的
    preprocessed_dir = 'preprocessed_data_librispeech'
    preprocessed_dir = preprocessed_dir_method(preprocessed_dir, args)
    if not os.path.exists(preprocessed_dir):
        os.makedirs(preprocessed_dir)

    if os.path.exists(preprocessed_dir) and os.listdir(preprocessed_dir):
        panduan = True
    else:
        panduan = False
    if not panduan:
        audio_feature_method(address_speakers, class_to_id, preprocessed_dir)
    data = AudioDataset(preprocessed_dir)
    trd, vad, ted = getlabeldataloader(args, data)
    print("-------------------------------")
    print("传入后的args.batch:{}".format(args.batch))

    print("args.num_classes的值是：{}".format(args.num_classes))
    return trd, vad, ted


def voxceleb1_data_method(args):
    spk_to_utts, speakers = voxceleb1_data(args.audio_file_address, args)
    class_to_id = classses(speakers)
    address_speakers = address_speakers_to_list(spk_to_utts, speakers)
    data = AudioDataset(data_dir=address_speakers, label_mapping=class_to_id)
    trd, vad, ted = getlabeldataloader(args, data)
    print("-------------------------------")
    print("传入后的args.batch:{}".format(args.batch))

    print("args.num_classes的值是：{}".format(args.num_classes))
    return trd, vad, ted


def voxceleb2_data_method(args):
    spk_to_utts, speakers = voxceleb2_data(args.audio_file_address, args)
    class_to_id = classses(speakers)
    address_speakers = address_speakers_to_list(spk_to_utts, speakers)
    # data = AudioDataset(data_dir=address_speakers, label_mapping=class_to_id)

    # 新加的
    preprocessed_dir = 'preprocessed_data_voxceleb2'
    if not os.path.exists(preprocessed_dir):
        os.makedirs(preprocessed_dir)


    if os.path.exists(preprocessed_dir) and os.listdir(preprocessed_dir):
        panduan = True
    else:
        panduan = False
    if not panduan:
        audio_feature_method(address_speakers, class_to_id, preprocessed_dir)
    data = AudioDataset(preprocessed_dir)

    trd, vad, ted = getlabeldataloader(args, data)
    print("-------------------------------")
    print("传入后的args.batch:{}".format(args.batch))

    print("args.num_classes的值是：{}".format(args.num_classes))
    return trd, vad, ted


def get_data(data_name):
    """返回具有给定名称的算法类。"""
    datalist = {'all': 'all_data', 'librispeech': 'librispeech_data_method',
                'VoxCeleb1': 'voxceleb1_data_method', 'Timit': 'timit_data_method',
                'Zhvoice': 'zhvoice_data_method', 'VoxCeleb2': 'voxceleb2_data_method'}
    if datalist[data_name] not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(data_name))
    return globals()[datalist[data_name]]


def librispeech_data(file_address, args):
    print("librispeech_data_to_text方法成功被调用")
    flac_files = glob.glob(os.path.join(file_address, "*", "*", "*.flac"))
    spk_to_utts = dict()
    speakers = []  # 唯一说话人标记的列表，为之后分组客户端用
    for flac_file in flac_files:
        basename = os.path.basename(flac_file)
        # 只截取最后文件名，限定说话人
        split_name = basename.split("-")
        spk = split_name[0]
        if spk not in spk_to_utts:
            spk_to_utts[spk] = [flac_file.replace('\\', '/')]
            speakers.append(split_name[0])
        else:
            spk_to_utts[spk].append(flac_file.replace('\\', '/'))

    # print(spk_to_utts)
    # print(speakers)
    args.num_classes = len(speakers)
    return spk_to_utts, speakers


def classses(speakerclasses):
    # 创建标签的映射，形成连续的标签，为之后训练做准备
    class_to_id = {label: i for i, label in enumerate(speakerclasses)}

    # if(len(class_to_id)==251):
    #     class_to_id.popitem()
    print(class_to_id)
    return class_to_id


def address_speakers_to_list(spk_to_utts, speakers):
    # 将数据集的集合转换成地址——说话人标签形式
    """
    示例函数，带有类型提示的参数。
    Args:
       spk_to_utts  (dict)
       speakers (list)
    Returns:
        address_speakers (list)
    """
    # print("spk_to_utts的值是：{}".format(spk_to_utts))
    address_speakers = []
    for i in range(len(speakers)):
        # print(spk_to_utts.get(speakers[i]))
        for j in range(len(spk_to_utts.get(speakers[i]))):
            list = []
            list.append(spk_to_utts.get(speakers[i])[j])
            list.append(speakers[i])
            address_speakers.append(list)
    # print("address_speakers的值是：{}".format(address_speakers))
    return address_speakers


def getlabeldataloader(args, data):
    trl, val, tel = getdataloader(args, data)
    # if args.alg == 'localmoml':
    #     return trl, val, tel
    trd, vad, ted = [], [], []
    for i in range(len(trl)):
        trd.append(torch.utils.data.DataLoader(
            trl[i], batch_size=args.batch, shuffle=True))
        vad.append(torch.utils.data.DataLoader(
            val[i], batch_size=args.batch, shuffle=False))
        ted.append(torch.utils.data.DataLoader(
            tel[i], batch_size=args.batch, shuffle=False))
    return trd, vad, ted


# 不知道
def get_whole_dataset(data_name):
    datalist = {'officehome': 'img_union_w', 'pacs': 'img_union_w', 'vlcs': 'img_union_w', 'medmnist': 'medmnist_w',
                'medmnistA': 'medmnist_w', 'medmnistC': 'medmnist_w', 'pamap': 'pamap_w', 'covid': 'covid_w'}
    if datalist[data_name] not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(data_name))
    return globals()[datalist[data_name]]


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # method = get_data('librispeech')
    # spk_to_utts, speakers = method("E:/VR-FL-project/firstproject/data/test-clean/LibriSpeech/test-clean")
    # class_to_id = classses(speakers)
    # # class_num = len(speakers)
    # address_speakers = address_speakers_to_list(spk_to_utts, speakers)
    # print("address_speakers_to_list方法调用成功")
    # data = AudioDataset(data_dir=address_speakers, label_mapping=class_to_id)
    # print("开始进行测试AudioDataset类：")
    # print(data.data)
    # print(data.labels)
    #
    # method = get_data('VoxCeleb1')
    # spk_to_utts, speakers = method("E:/VR-FL-project/firstproject/data/vox1_test_wav/wav")

    # timit_data("E:/VR-FL-project/firstproject/data/Timit/data/TIMIT/TRAIN")
    preprocessed_dir = 'preprocessed_data_zhvoice_zhthchs30'
    directory = "F:/dataset/zhvoice/zhthchs30"
    preprocessed_dir_method(preprocessed_dir,directory)