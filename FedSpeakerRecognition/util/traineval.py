import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datautil.datasplit import define_pretrain_dataset
from datautil.dataprepare import get_whole_dataset


def train(model, data_loader, optimizer, loss_fun, device):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
    # for i, (alldata) in enumerate(data_loader):
    #     data, target = alldata
        # 以上两行是修改的，可以删除
        # data = data.to(device).float()
        # target = target.to(device).long()
        data =data.cuda(non_blocking=True).float()
        target = target.cuda(non_blocking=True).long()
        optimizer.zero_grad()  # 自己加上去的

        # print("开始检查数据维度")
        # print("Tensor Shape:", data.shape)
        # shape = data.shape
        # print("Tensor Shape (as tuple):", shape)
        # # 获取Tensor的维度数量
        # num_dimensions = data.ndim
        # print("Number of Dimensions:", num_dimensions)
        # # 获取Tensor的各个维度的大小
        # sizes = data.size()
        # print("Sizes of Dimensions:", sizes)

        # # 在第一个维度上添加一个额外的维度
        # input_tensor = data.unsqueeze(1)

        # # 现在input_tensor的形状将是 [1, 32, 100, 40]
        # print("Updated Tensor Shape:", input_tensor.shape)

        output = model(data)
        # print("目标值:", target)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        # pred = output.data.max(1)[1]
        # correct += pred.eq(target.view(-1)).sum().item()

        # 下面两行自己加的
        _, predicted = torch.max(output, 1)
        correct += (predicted == target).sum().item()

        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_all / len(data_loader), correct / total


def test(model, data_loader, loss_fun, device):
    model.eval()
    loss_all = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device).float()
            target = target.to(device).long()
            output = model(data)
            loss = loss_fun(output, target)
            loss_all += loss.item()
            total += target.size(0)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()

        return loss_all / len(data_loader), correct / total


def train_prox(args, model, server_model, data_loader, optimizer, loss_fun, device):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    for step, (data, target) in enumerate(data_loader):
        data = data.to(device).float()
        target = target.to(device).long()
        output = model(data)
        loss = loss_fun(output, target)
        if step > 0:
            w_diff = torch.tensor(0., device=device)
            for w, w_t in zip(server_model.parameters(), model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)

            w_diff = torch.sqrt(w_diff)
            loss += args.mu / 2. * w_diff

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct / total


def trainwithteacher(model, data_loader, optimizer, loss_fun, device, tmodel, lam, args, flag):
    model.train()
    if tmodel:  # 教师模型 tmodel
        tmodel.eval()
        if not flag:
            with torch.no_grad():
                for key in tmodel.state_dict().keys():
                    if 'num_batches_tracked' in key:
                        pass
                    elif args.nosharebn and 'bn' in key:
                        pass
                    else:
                        model.state_dict()[key].data.copy_(
                            tmodel.state_dict()[key])
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        optimizer.zero_grad()

        data = data.to(device).float()
        target = target.to(device).long()
        output = model(data)
        f1 = model.get_sel_fea(data, args.plan)
        loss = loss_fun(output, target)
        if flag and tmodel:
            f2 = tmodel.get_sel_fea(data, args.plan).detach()
            loss += (lam * F.mse_loss(f1, f2))
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]  # 模型的输出中获取预测的类别标签。
        correct += pred.eq(target.view(-1)).sum().item()  # 计算正确预测的样本数量

        loss.backward()
        optimizer.step()

    return loss_all / len(data_loader), correct / total


def pretrain_model(args, model, filename, device='cuda'):
    print('===training pretrained model===')
    data = get_whole_dataset(args.dataset)(args)
    predata = define_pretrain_dataset(args, data)
    traindata = torch.utils.data.DataLoader(
        predata, batch_size=args.batch, shuffle=True)
    loss_fun = nn.CrossEntropyLoss()
    opt = optim.SGD(params=model.parameters(), lr=args.lr)
    for _ in range(args.pretrained_iters):
        _, acc = train(model, traindata, opt, loss_fun, device)
    torch.save({
        'state': model.state_dict(),
        'acc': acc
    }, filename)
    print('===done!===')
