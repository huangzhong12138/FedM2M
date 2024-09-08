import torch
import torch.nn as nn
import torch.nn.functional as func
from collections import OrderedDict
import torch.nn.functional as F
from network.ecapatdnn import *
from network.pooling import *
import math


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                ('bn2', nn.BatchNorm2d(192)),
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                ('bn3', nn.BatchNorm2d(384)),
                ('relu3', nn.ReLU(inplace=True)),

                ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
                ('bn4', nn.BatchNorm2d(256)),
                ('relu4', nn.ReLU(inplace=True)),

                ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('bn5', nn.BatchNorm2d(256)),
                ('relu5', nn.ReLU(inplace=True)),
                ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
            ])
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(256 * 6 * 6, 4096)),
                ('bn6', nn.BatchNorm1d(4096)),
                ('relu6', nn.ReLU(inplace=True)),

                ('fc2', nn.Linear(4096, 4096)),
                ('bn7', nn.BatchNorm1d(4096)),
                ('relu7', nn.ReLU(inplace=True)),

                ('fc3', nn.Linear(4096, num_classes)),
            ])
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def getallfea(self, x):
        fealist = []
        for i in range(len(self.features)):
            if i in [1, 5, 9, 12, 15]:
                fealist.append(x.clone().detach())
            x = self.features[i](x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        for i in range(len(self.classifier)):
            if i in [1, 4]:
                fealist.append(x.clone().detach())
            x = self.classifier[i](x)
        return fealist

    def getfinalfea(self, x):
        for i in range(len(self.features)):
            x = self.features[i](x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        for i in range(len(self.classifier)):
            if i == 6:
                return [x]
            x = self.classifier[i](x)
        return x

    def get_sel_fea(self, x, plan=0):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if plan == 0:
            y = x
        elif plan == 1:
            y = self.classifier[5](self.classifier[4](self.classifier[3](
                self.classifier[2](self.classifier[1](self.classifier[0](x))))))
        else:
            y = []
            y.append(x)
            x = self.classifier[2](self.classifier[1](self.classifier[0](x)))
            y.append(x)
            x = self.classifier[5](self.classifier[4](self.classifier[3](x)))
            y.append(x)
            y = torch.cat(y, dim=1)
        return y


class PamapModel(nn.Module):
    def __init__(self, n_feature=64, out_dim=10):
        super(PamapModel, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=27, out_channels=16, kernel_size=(1, 9))
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(1, 9))
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        self.fc1 = nn.Linear(in_features=32 * 44, out_features=n_feature)
        self.fc1_relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=n_feature, out_features=out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(self.relu1(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool2(self.relu2(self.bn2(x)))
        x = x.reshape(-1, 32 * 44)
        feature = self.fc1_relu(self.fc1(x))
        out = self.fc2(feature)
        return out

    def getallfea(self, x):
        fealist = []
        x = self.conv1(x)
        fealist.append(x.clone().detach())
        x = self.pool1(self.relu1(self.bn1(x)))
        x = self.conv2(x)
        fealist.append(x.clone().detach())
        return fealist

    def getfinalfea(self, x):
        x = self.conv1(x)
        x = self.pool1(self.relu1(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool2(self.relu2(self.bn2(x)))
        x = x.reshape(-1, 32 * 44)
        feature = self.fc1_relu(self.fc1(x))
        return [feature]

    def get_sel_fea(self, x, plan=0):
        if plan == 0:
            x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            x = x.reshape(-1, 32 * 44)
            fealist = x
        elif plan == 1:
            x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            x = x.reshape(-1, 32 * 44)
            feature = self.fc1_relu(self.fc1(x))
            fealist = feature
        else:
            fealist = []
            x = self.conv1(x)
            x = self.pool1(self.relu1(self.bn1(x)))
            fealist.append(x.view(x.shape[0], -1))
            x = self.conv2(x)
            x = self.pool2(self.relu2(self.bn2(x)))
            fealist.append(x.view(x.shape[0], -1))
            x = x.reshape(-1, 32 * 44)
            feature = self.fc1_relu(self.fc1(x))
            fealist.append(feature)
            fealist = torch.cat(fealist, dim=1)
        return fealist


class lenet5v(nn.Module):
    def __init__(self):
        super(lenet5v, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 11)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        return y

    def getallfea(self, x):
        fealist = []
        y = self.conv1(x)
        fealist.append(y.clone().detach())
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        fealist.append(y.clone().detach())
        return fealist

    def getfinalfea(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        return [y]

    def get_sel_fea(self, x, plan=0):
        if plan == 0:
            x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            x = x.view(x.shape[0], -1)
            fealist = x
        elif plan == 1:
            x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            x = x.view(x.shape[0], -1)
            x = self.relu3(self.fc1(x))
            x = self.relu4(self.fc2(x))
            fealist = x
        else:
            fealist = []
            x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            x = x.view(x.shape[0], -1)
            fealist.append(x)
            x = self.relu3(self.fc1(x))
            fealist.append(x)
            x = self.relu4(self.fc2(x))
            fealist.append(x)
            fealist = torch.cat(fealist, dim=1)
        return fealist


class VoiceprintLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(VoiceprintLSTM, self).__init__()

        # # 三层LSTM层
        self.num_layers = num_layers  # 添加这一行
        self.hidden_size = hidden_size
        bidirectional = True
        # self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        # self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        # self.lstm3 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=0.2)

        # 这是另外的
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.fc = nn.Linear(hidden_size, num_classes)

        # # 全连接层
        # self.fc = nn.Linear(hidden_size, num_classes)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Output layer
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)

    def forward(self, x):

        # 这是新的神经网络模型
        # Initialize LSTM hidden and cell states
        D = 2 if self.lstm.bidirectional else 1
        h0 = torch.zeros(
            D * self.lstm.num_layers, x.shape[0], self.lstm.hidden_size
        ).cuda()
        c0 = torch.zeros(
            D * self.lstm.num_layers, x.shape[0], self.lstm.hidden_size
        ).cuda()

        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Aggregate frames (e.g., mean pooling)
        aggregated_output = torch.mean(lstm_out, dim=1, keepdim=False)

        # Final classification layer
        output = self.fc(aggregated_output)

        return output

        # # 初始化LSTM的隐藏状态
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # # Forward pass through LSTM layers
        # out, _ = self.lstm1(x, (h0, c0))
        # out, _ = self.lstm2(out, (h0, c0))
        # out, _ = self.lstm3(out, (h0, c0))
        # # # 前向传播，获取LSTM的输出
        # # out, _ = self.lstm(x, (h0, c0))
        # # 取最后一个时间步的输出作为特征向量
        # out = out[:, -1, :]
        # # 通过全连接层进行分类
        # out = self.fc(out)
        # return out

    def get_sel_fea(self, x, plan=0):
        # 在这里实现获取选择性特征的逻辑
        # 根据 plan 参数决定不同的特征选择方式
        if plan == 0:
            # 方案 0 的特征选择逻辑
            D = 2 if self.lstm.bidirectional else 1
            h0 = torch.zeros(
                D * self.lstm.num_layers, x.shape[0], self.lstm.hidden_size
            ).cuda()
            c0 = torch.zeros(
                D * self.lstm.num_layers, x.shape[0], self.lstm.hidden_size
            ).cuda()

            lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
            # 获取第一层LSTM输出
            first_layer_output = lstm_out[:, :, :self.hidden_size]
            listfea = first_layer_output

        elif plan == 1:
            # 方案 1 的特征选择逻辑
            D = 2 if self.lstm.bidirectional else 1
            h0 = torch.zeros(
                D * self.lstm.num_layers, x.shape[0], self.lstm.hidden_size
            ).cuda()
            c0 = torch.zeros(
                D * self.lstm.num_layers, x.shape[0], self.lstm.hidden_size
            ).cuda()

            lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
            # 获取第二层LSTM输出
            second_layer_output = lstm_out[:, :, self.hidden_size:2 * self.hidden_size]
            listfea = second_layer_output


        else:
            D = 2 if self.lstm.bidirectional else 1
            h0 = torch.zeros(
                D * self.lstm.num_layers, x.shape[0], self.lstm.hidden_size
            ).cuda()
            c0 = torch.zeros(
                D * self.lstm.num_layers, x.shape[0], self.lstm.hidden_size
            ).cuda()

            lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
            # 获取第三层 LSTM 的输出
            third_layer_output = lstm_out[:, :, 2 * self.hidden_size:]
            listfea = third_layer_output

        return listfea


# ecapa-tdnn
class EcapaTdnn(nn.Module):
    def __init__(self, input_size=40, channels=512, embd_dim=192, pooling_type="ASP"):
        super().__init__()
        # dropout_prob = 0.1
        self.layer1 = Conv1dReluBn(input_size, channels, kernel_size=5, padding=2, dilation=1)
        self.layer2 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=4)
        self.layer3 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=4)
        self.layer4 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=4)
        # self.dropout = nn.Dropout(dropout_prob)  # 添加dropout层

        cat_channels = channels * 3
        self.emb_size = embd_dim
        self.conv = nn.Conv1d(cat_channels, cat_channels, kernel_size=1)
        if pooling_type == "ASP":
            self.pooling = AttentiveStatsPool(cat_channels, 128)
            self.bn1 = nn.BatchNorm1d(cat_channels * 2)
            self.linear = nn.Linear(cat_channels * 2, embd_dim)
            self.bn2 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == "SAP":
            self.pooling = SelfAttentivePooling(cat_channels, 128)
            self.bn1 = nn.BatchNorm1d(cat_channels)
            self.linear = nn.Linear(cat_channels, embd_dim)
            self.bn2 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == "TAP":
            self.pooling = TemporalAveragePooling()
            self.bn1 = nn.BatchNorm1d(cat_channels)
            self.linear = nn.Linear(cat_channels, embd_dim)
            self.bn2 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == "TSP":
            self.pooling = TemporalStatisticsPooling()
            self.bn1 = nn.BatchNorm1d(cat_channels * 2)
            self.linear = nn.Linear(cat_channels * 2, embd_dim)
            self.bn2 = nn.BatchNorm1d(embd_dim)
        else:
            raise Exception(f'没有{pooling_type}池化层！')

    def forward(self, x):
        """
        Compute embeddings.

        Args:
            x (torch.Tensor): Input data with shape (N, time, freq).

        Returns:
            torch.Tensor: Output embeddings with shape (N, self.emb_size, 1)
        """
        x = x.transpose(2, 1)
        out1 = self.layer1(x)
        out2 = self.layer2(out1) + out1
        out3 = self.layer3(out1 + out2) + out1 + out2
        out4 = self.layer4(out1 + out2 + out3) + out1 + out2 + out3

        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        # out = self.dropout(out)  # 应用dropout(试试看)
        out = self.bn1(self.pooling(out))
        # out = self.dropout(out)  # 应用dropout(试试看)
        out = self.bn2(self.linear(out))
        return out

    def get_sel_fea(self, x, plan=0):

        if plan == 0 or plan == 1:
            x = x.transpose(2, 1)
            out1 = self.layer1(x)
            out2 = self.layer2(out1) + out1
            out3 = self.layer3(out1 + out2) + out1 + out2
            out4 = self.layer4(out1 + out2 + out3) + out1 + out2 + out3

            out = torch.cat([out2, out3, out4], dim=1)
            out = F.relu(self.conv(out))
            out = self.bn1(self.pooling(out))
            out = self.bn2(self.linear(out))
            listfea = out

        return listfea


class TDNN(nn.Module):
    def __init__(self, input_size=40, channels=512, embd_dim=400, pooling_type="ASP"):
        super(TDNN, self).__init__()
        self.emb_size = embd_dim
        self.td_layer1 = torch.nn.Conv1d(in_channels=input_size, out_channels=512, dilation=1, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm1d(512)
        self.td_layer2 = torch.nn.Conv1d(in_channels=512, out_channels=512, dilation=2, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm1d(512)
        self.td_layer3 = torch.nn.Conv1d(in_channels=512, out_channels=512, dilation=3, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm1d(512)
        self.td_layer4 = torch.nn.Conv1d(in_channels=512, out_channels=512, dilation=1, kernel_size=1, stride=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.td_layer5 = torch.nn.Conv1d(in_channels=512, out_channels=channels, dilation=1, kernel_size=1, stride=1)

        if pooling_type == "ASP":
            self.pooling = AttentiveStatsPool(channels, 128)
            self.bn5 = nn.BatchNorm1d(channels * 2)
            self.linear = nn.Linear(channels * 2, embd_dim)
            self.bn6 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == "SAP":
            self.pooling = SelfAttentivePooling(channels, 128)
            self.bn5 = nn.BatchNorm1d(channels)
            self.linear = nn.Linear(channels, embd_dim)
            self.bn6 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == "TAP":
            self.pooling = TemporalAveragePooling()
            self.bn5 = nn.BatchNorm1d(channels)
            self.linear = nn.Linear(channels, embd_dim)
            self.bn6 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == "TSP":
            self.pooling = TemporalStatisticsPooling()
            self.bn5 = nn.BatchNorm1d(channels * 2)
            self.linear = nn.Linear(channels * 2, embd_dim)
            self.bn6 = nn.BatchNorm1d(embd_dim)
        else:
            raise Exception(f'没有{pooling_type}池化层！')

    def forward(self, x):
        """
        Compute embeddings.

        Args:
            x (torch.Tensor): Input data with shape (N, time, freq).

        Returns:
            torch.Tensor: Output embeddings with shape (N, self.emb_size, 1)
        """
        x = x.transpose(2, 1)
        x = F.relu(self.td_layer1(x))
        x = self.bn1(x)
        x = F.relu(self.td_layer2(x))
        x = self.bn2(x)
        x = F.relu(self.td_layer3(x))
        x = self.bn3(x)
        x = F.relu(self.td_layer4(x))
        x = self.bn4(x)
        x = F.relu(self.td_layer5(x))
        out = self.bn5(self.pooling(x))
        out = self.bn6(self.linear(out))
        return out


#####resnet


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, in_channels=1):
        self.inplanes = 32
        m_channels = 32
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, m_channels, layers[0])
        self.layer2 = self._make_layer(block, m_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, m_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, m_channels * 8, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(1, stride=1)
        print("block.expansion的值是：{}".format(block.expansion))
        self.fc = nn.Linear(m_channels * 8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = x.transpose(1, 0)
        x = x.unsqueeze(0)
        x = x.transpose(0, 1)
        sizes = x.size()
        print("模型转变后Sizes of Dimensions:", sizes)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(classes):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], classes)
