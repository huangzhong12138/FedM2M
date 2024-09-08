import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from alg.core.comm import communication
from util.traineval import train, test

from alg.fedavg import fedavg
from util.modelsel import modelsel
import torch.nn.functional as F


class localmoml(torch.nn.Module):
    def __init__(self, args):
        super(localmoml, self).__init__()
        self.server_model, self.client_model, self.client_weight = modelsel(
            args, args.device)
        self.u_weights = self.client_model
        self.optimizers = [optim.SGD(params=self.client_model[idx].parameters(
        ), lr=args.lr) for idx in range(args.n_clients)]
        self.loss_fun = nn.CrossEntropyLoss()
        self.args = args

    def local_train(self, c_idx, dataloader, epoch):
        # print("local_train方法开始执行，第{}个客户端开始".format(c_idx))
        # print(f"local_train method called with c_idx={c_idx}, epoch={epoch}")
        train_loss, train_acc, u_weights = client_train(
            self.client_model[c_idx], dataloader, self.optimizers[c_idx], self.loss_fun, self.args.device,
            self.args.update_lr)
        for param, u_weight in zip(self.client_model[c_idx].parameters(), u_weights):
            param.data = u_weight
        # print("local_train方法开始执行，第{}个客户端结束".format(c_idx))
        # print("=========================================================")
        return train_loss, train_acc

    def local_train_moml(self, c_idx, query_loader, val_loader):
        # print("local_train_moml方法开始执行")
        client_train_query(self.client_model[c_idx], query_loader, val_loader, self.optimizers[c_idx], self.loss_fun,
                           self.args.update_lr, self.args.beta, self.u_weights[c_idx], self.args.device,
                           self.args.meta_lr)

    def server_aggre(self):
        self.server_model, self.client_model = communication(
            self.args, self.server_model, self.client_model, self.client_weight)

    def client_eval(self, c_idx, dataloader):
        train_loss, train_acc = test(
            self.client_model[c_idx], dataloader, self.loss_fun, self.args.device)
        return train_loss, train_acc

def client_train(model, train_loader, optimizer, loss_fun, device, update_lr):
    # 这是联邦元学习的本地训练，直接写在这里，暂时不集成
    # 全局模型和客户端模型都已经拷贝成功，无需复制重新加载
    # c_idx代表客户端编号
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    # train_loader是剩下的训练集
    for data, target in train_loader:
        data = data.to(device).float()
        target = target.to(device).long()
        optimizer.zero_grad()  # 自己加上去的
        output = model(data)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        # 以上是模型训练
        _, predicted = torch.max(output, 1)
        correct += (predicted == target).sum().item()

        # loss.backward()
        # optimizer.step()元学习要求手动进行梯度调整
        w_grads = torch.autograd.grad(loss, model.parameters())
        u_weights = list(map(lambda p: p[0].data - update_lr * p[1].data, zip(model.parameters(), w_grads)))

    return loss_all / len(train_loader), correct / total, u_weights


def client_train_query(model, query_loader, val_loaders, optimizer, loss_fun, update_lr, beta, u_weights, device,
                       meta_lr):
    iter_train_acc_list = []
    iter_loss_list = []
    model.train()
    loss_all = 0
    loss_all_query = 0
    total = 0
    total_query = 0
    correct = 0
    correct_query = 0
    for data, target in query_loader:
        data = data.to(device).float()
        target = target.to(device).long()
        optimizer.zero_grad()  # 自己加上去的
        output = model(data)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)

        _, predicted = torch.max(output, 1)
        correct += (predicted == target).sum().item()

    w_grads = torch.autograd.grad(loss, model.parameters())  # 计算损失相对于模型参数的梯度。
    # 将模型参数转换为列表
    model_params = list(model.parameters())
    # fast_weights = list(map(lambda p: p[0].data - update_lr * p[1].data, zip(model.parameters(), w_grads)))
    fast_weights = list(map(lambda p: p[0].data - update_lr * p[1].data, zip(model_params, w_grads)))
    # 根据一定的学习率update_lr更新模型的参数得到fast_weights。
    # print("开始进行报错检测")
    # u_weights = [u_weights]
    # print("这已经是u_weights = [u_weights]后的参数类型了")
    # print(type(u_weights))
    # print(type(fast_weights))
    # u_weights = list(map(lambda p: p[0].data * (1 - beta) + beta * p[1].data, zip(u_weights, fast_weights)))
    # 使用u_weights和fast_weights的加权平均更新模型的参数。
    for param_u, param_fast in zip(u_weights.parameters(), fast_weights):
        param_u.data = param_u.data * (1 - beta) + beta * param_fast.data

    # compute gradients w.r.t "u"
    model_w_copy = copy.deepcopy(model.state_dict())
    for w, u in zip(model.parameters(), u_weights.parameters()):
        w.data = u.data

    pre_val = []
    # for data, target in range(val_loaders):
    for idx, (data, target) in enumerate(val_loaders):
        data = data.to(device).float()
        target = target.to(device).long()
        optimizer.zero_grad()  # 自己加上去的
        output = model(data)
        pre_val.append(output)  # 这里是新加的，好像没用
        loss = loss_fun(output, target)
        loss_all_query += loss.item()
        total_query += target.size(0)

        _, predicted = torch.max(output, 1)
        correct_query += (predicted == target).sum().item()
        iter_train_acc_list.append(correct_query)

    grads_q = torch.autograd.grad(loss, model.parameters())
    # change back previous "w" an update local w
    # 改回之前的“w”并更新本地 w
    for name, var in model.named_parameters():
        var.data = model_w_copy[name].data
        # 将模型参数恢复到之前备份的状态，即将模型参数重置为上一个循环中更新之前的值。
    for var, grad in zip(model.parameters(), grads_q):
        var.data = var.data - meta_lr * grad.data
        # 使用学习率 meta_lr 对模型参数进行更新。
    iter_loss_list.append(loss.item())  # 当前循环中计算得到的损失值添加到 iter_loss_list 列表中


def one_to_supportandquery(dataloader):
    # 步骤 1: 将 DataLoader 转换为 Python 列表
    if isinstance(dataloader, torch.utils.data.DataLoader):
        # all_samples = list(dataloader)
        original_dataset = dataloader.dataset
        all_samples = len(original_dataset)
        print("dataloader数据长度是：{}".format(all_samples))
        # 步骤 2: 划分支持集和查询集
        # 例如，这里使用 PyTorch 的 random_split 函数来划分成两个子集
        # support_set_size = int(0.2 * len(all_samples))  # 20% 用于支持集
        # support_set, query_set = random_split(all_samples, [support_set_size, len(all_samples) - support_set_size])
        #
        # # 步骤 3: 创建两个新的 DataLoader
        # batch_size = dataloader.batch_size
        # support_set_loader = DataLoader(support_set, batch_size=batch_size, shuffle=True)
        # query_set_loader = DataLoader(query_set, batch_size=batch_size, shuffle=True)

        num_support = int(all_samples * 0.2)
        num_query = all_samples - num_support
        support_set_loader, query_set_loader = random_split(original_dataset, [num_support, num_query])

        batch_size = dataloader.batch_size
        support_set_loader = DataLoader(support_set_loader, batch_size=batch_size, shuffle=True)
        query_set_loader = DataLoader(query_set_loader, batch_size=batch_size, shuffle=True)

        # 输出元学习的支持集和查询集
        return support_set_loader, query_set_loader
    else:
        print("one_to_supportandquery报错")
        raise ValueError("Invalid input")
