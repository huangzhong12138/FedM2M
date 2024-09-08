import numpy as np
import torch
from metrics import TprAtFpr
from tqdm import tqdm

def evaluate_eer(model,dataloader,device):
    model.eval()
    features, labels = None, None  # 初始化存放特征和标签的变量

    with torch.no_grad():
        for data, target in dataloader:
            # 将音频数据和标签转移到指定设备（GPU或CPU）
            data = data.to(device).float()
            target = target.to(device).long()

            feature = model.backbone(data).data.cpu().numpy() # 获取特征并转为numpy数组
            label = label.data.cpu().numpy()
            # 存放特征
            features = np.concatenate((features, feature)) if features is not None else feature
            labels = np.concatenate((labels, label)) if labels is not None else label

    model.train()
    metric = TprAtFpr()
    labels = labels.astype(np.int32)
    print('开始两两对比音频特征...')
    for i in tqdm(range(len(features))):
        feature_1 = features[i]
        feature_1 = np.expand_dims(feature_1, 0).repeat(len(features) - i, axis=0)
        feature_2 = features[i:]
        feature_1 = torch.tensor(feature_1, dtype=torch.float32)
        feature_2 = torch.tensor(feature_2, dtype=torch.float32)
        score = torch.nn.functional.cosine_similarity(feature_1, feature_2, dim=-1).data.cpu().numpy().tolist()
        y_true = np.array(labels[i] == labels[i:]).astype(np.int32).tolist()
        metric.add(y_true, score)
    tprs, fprs, thresholds, eer, index = metric.calculate()
    tpr, fpr, threshold = tprs[index], fprs[index], thresholds[index]
    return tpr, fpr, eer, threshold