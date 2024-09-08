import argparse
import os

import numpy as np
import torch
from datautil.dataprepare import *
from alg import algs
from util.evalandprint import evalandprint
from util.config import *
from alg.localmoml import *
from alg.localmoml import localmoml as Localmoml

if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default='localmoml',
                        help='Algorithm to choose: [base | fedavg | metafed ]')
    parser.add_argument('--dataset', type=str, default='Zhvoice',
                        help='[vlcs | pacs | officehome | pamap | covid | Zhvoice| librispeech ]')
    parser.add_argument('--audio_file_address', type=str,
                        default='F:/dataset/zhstcmds_shengxia',
                        help='data percent to use')
    parser.add_argument('--save_path', type=str,
                        default='./cks/', help='path to save the checkpoint')
    parser.add_argument('--n_clients', type=int,
                        default=10, help='number of clients')
    parser.add_argument('--non_iid_alpha', type=float,
                        default=1, help='data split for label shift')
    parser.add_argument('--partition_data', type=str,
                        default='non_iid_dirichlet', help='partition data way')
    parser.add_argument('--iters', type=int, default=600,
                        help='iterations for communication')
    parser.add_argument('--device', type=str,
                        default='cuda', help='[cuda | cpu]')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch', type=int, default=512, help='batch size')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--wk_iters', type=int, default=20,
                        help='optimization iters in local worker between communication')
    parser.add_argument('--nosharebn', action='store_true',
                        help='not share bn')
    parser.add_argument('--plan', type=int,
                        default=0, help='choose the feature type')
    parser.add_argument('--pretrained_iters', type=int,
                        default=150, help='iterations for pretrained models')
    parser.add_argument('--saved_data', type=bool,
                        default=True, help='Decide whether to record data for drawing')

    # 这是挑选客户端个数操作
    parser.add_argument('--k', type=int,
                        default=8, help='tiaoxuankehuduan')
    # 元算法特定参数
    parser.add_argument('--update_lr', type=float, default=0.02,
                        help='localmoml_canshu')
    parser.add_argument('--beta', type=float, default=0.8,
                        help='localmoml_canshu')
    parser.add_argument('--meta_lr', type=float, default=0.02,
                        help='localmoml_canshu')
    parser.add_argument('--H', type=int, default=20,
                        help='localmoml_canshu')
    args = parser.parse_args()

    args.random_state = np.random.RandomState(1)
    # 设置随机数生成器的全局种子
    set_random_seed(args.seed)

    # print(args.audio_file_address)
    # print(args.alg)
    if args.device == 'cuda':
        # 使用 GPU 进行计算
        device = torch.device('cuda')
    else:
        # 使用 CPU 进行计算
        device = torch.device('cpu')

    train_loaders, val_loaders, test_loaders = get_data(args.dataset)(args)  # 数据加载器

    algclass = algs.get_algorithm_class(args.alg)(args)

    # 是否发生了最佳性能变化
    best_changed = False

    best_acc = [0] * args.n_clients
    best_tacc = [0] * args.n_clients
    start_iter = 0
    SAVE_PATH = os.path.join(args.save_path, args.alg)
    # 初始化空列表来存储每一轮的损失和准确率
    train_average_loss_history = []
    train_average_acc_history = []
    val_average_loss_history = []
    val_average_acc_history = []
    test_average_acc_history = []



    # localmoml = Localmoml(args)
    for epoch in range(args.iters):
        print(f"============ Train round {epoch} ============")
        # model_weigths = [torch.FloatTensor(m).cuda() for m in model_pools[uid[i].numpy()[0]]]
        # model_dicts = dict(zip(model_names, model_weigths))  加载预训练模型
        #
        iter_train_acc_list = []
        iter_loss_list = []

        # 每次训练从clients列表中随机抽取k个进行训练。candidates也是列表，里面有5个客户端实例
        my_array = list(range(args.n_clients))
        print("sorted(args.n_clients)值是不是：{}".format(my_array))
        candidates = random.sample(my_array, args.k)
        print("selected clients are: ")
        for c in candidates:
            print('client_id: ', c)


        for c_idx in candidates:
            support_set_loader, query_set_loader = one_to_supportandquery(train_loaders[c_idx])
            # print(f"Before local_train, client_idx,epoch: {c_idx, epoch}")
            for _ in range(1):
                algclass.local_train(c_idx, support_set_loader, epoch)

            for h in range(args.H):  # 元学习过程，这里的轮次稍后设置args
                # print("第几轮运行：{}".format(h))
                algclass.local_train_moml(c_idx, query_set_loader, val_loaders[c_idx])
        # server aggregation
        algclass.server_aggre()

        best_acc, best_tacc, best_changed, train_average_loss_history, train_average_acc_history, val_average_loss_history, val_average_acc_history, test_average_acc_history = evalandprint(
            args, algclass, train_loaders, val_loaders, test_loaders,  SAVE_PATH, best_acc, best_tacc, epoch,
            best_changed,
            train_average_loss_history,
            train_average_acc_history,
            val_average_loss_history,
            val_average_acc_history,
            test_average_acc_history)  # 用于评估模型性能并记录最佳性能的变化

    s = 'Personalized test acc for each client: '
    for item in best_tacc:
        s += f'{item:.4f},'
    mean_acc_test = np.mean(np.array(best_tacc))
    s += f'\nAverage accuracy: {mean_acc_test:.4f}'
    print(s)
    # print("train_average_loss_history的值是：{}".format(train_average_loss_history))

    # 开始进行保存数据文件
    import json

    if args.saved_data:
        root_result = './zhstcmds_shengxia_ceshi_XXX/'
        os.makedirs(root_result, exist_ok=True)
        os.makedirs(root_result + args.dataset, exist_ok=True)
        file = root_result + args.dataset + '/' + str(args.n_clients)
        os.makedirs(file, exist_ok=True)
        file = file + '/' + str(args.non_iid_alpha) + '.json'
        if not os.path.exists(file):
            record_data = {
                'train_average_loss_history': train_average_loss_history,
                'train_average_acc_history': train_average_acc_history,
                'val_average_loss_history': val_average_loss_history,
                'val_average_acc_history': val_average_acc_history,
                'test_average_acc_history': test_average_acc_history
            }
            with open(file, 'w') as f:
                json.dump(record_data, f)

        print("开始保存模型数据......")
        if args.audio_file_address == "F:/dataset/zhvoice/zhthchs30":
            dataset_name = "zhthchs30"+"tosave"
        elif args.audio_file_address == 'F:/dataset/zhstcmds_shengxia':
            dataset_name = "zhstcmds_shengxia"+"tosave"
        else:
            dataset_name = args.dataset+"tosave"
        SAVE_PATH = os.path.join(dataset_name, args.alg,str(args.non_iid_alpha),'checkpoint.pth')
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        print(f' Saving the local and server checkpoint to {SAVE_PATH}')
        # 检查并删除旧模型文件
        if os.path.exists(SAVE_PATH):
            os.remove(SAVE_PATH)
            print(f'Removing old model file: {SAVE_PATH}')
        tosave = {}
        for i, tmodel in enumerate(algclass.client_model):
            tosave['client_model_' + str(i)] = tmodel.state_dict()
        tosave['server_model'] = algclass.server_model.state_dict()
        torch.save(tosave, SAVE_PATH)