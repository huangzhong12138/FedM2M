import argparse
import os

import numpy as np
from datautil.dataprepare import *
from alg import algs
from util.evalandprint import evalandprint
from util.config import *

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='Algorithm to choose: [base | fedavg | metafed | fedprox |fedbn]')
    parser.add_argument('--dataset', type=str, default='Zhvoice',
                        help='[librispeech | VoxCeleb1 | Timit| Zhvoice |VoxCeleb2]')
    parser.add_argument('--audio_file_address', type=str,
                        default="F:/dataset/zhvoice/zhthchs30",
                        help='data percent to use')
    # "E:/VR-FL-project/firstproject/data/vox1_test_wav/wav",
    # E:/VR-FL-project/firstproject/data/train-clean-100/train-clean-100/LibriSpeech/train-clean-100
    # "E:/VR-FL-project/firstproject/data/test-clean/LibriSpeech/test-clean"
    # "F:/dataset/zhvoice/zhthchs30"
    # "F:/dataset/vox2_test_aac/aac"
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
                        default=1, help='choose the feature type')
    parser.add_argument('--pretrained_iters', type=int,
                        default=1, help='iterations for pretrained models')
    parser.add_argument('--saved_data', type=bool,
                        default=True, help='Decide whether to record data for drawing')
    # 这是挑选客户端个数操作
    parser.add_argument('--k', type=int,
                        default=8, help='tiaoxuankehuduan')

    # 这是联邦变分噪声模块，true代表使用
    parser.add_argument('--FVN', type=bool,
                        default=False, help='Decide whether Federated Variational Noise')

    # 算法特定参数
    parser.add_argument('--mu', type=float, default=1e-3,
                        help='The hyper parameter for fedprox')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='threshold to use copy or distillation, hyperparmeter for metafed')
    parser.add_argument('--lam', type=float, default=1.0,
                        help='init lam, hyperparmeter for metafed')
    parser.add_argument('--model_momentum', type=float,
                        default=0.5, help='hyperparameter for fedap')
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
    # for i in range(args.n_clients):
    #     print(len(test_loaders[i].dataset))

    algclass = algs.get_algorithm_class(args.alg)(args)

    if args.alg == 'fedap':
        algclass.set_client_weight(train_loaders)
    elif args.alg == 'metafed':
        algclass.init_model_flag(train_loaders, val_loaders)
        args.iters = args.iters - 1
        print('Common knowledge accumulation stage')  # 常识积累阶段
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

    for a_iter in range(start_iter, args.iters):
        print(f"============ Train round {a_iter} ============")
        # 每次训练从clients列表中随机抽取k个进行训练。candidates也是列表，里面有5个客户端实例
        my_array = list(range(args.n_clients))
        print("sorted(args.n_clients)值是不是：{}".format(my_array))
        candidates = random.sample(my_array, args.k)
        print("selected clients are: ")
        for c in candidates:
            print('client_id: ', c)

        # 根据FVN参数决定是否引入fvn模块
        if args.FVN:
            for client_idx in candidates:
                algclass.fvn(client_idx, args.lr * 0.001)

        if args.alg == 'metafed':
            # for c_idx in range(args.n_clients):  # 循环遍历所有客户端。
            for c_idx in candidates:
                algclass.client_train(
                    c_idx, train_loaders[algclass.csort[c_idx]], a_iter)  # 迭代轮次（a_iter）参数
            algclass.update_flag(val_loaders)
            # 用于更新客户端的标志。这个操作可能涉及到评估客户端的性能，以确定是否发生了最佳性能变化。
        else:
            # local client training（原代码）
            for wi in range(args.wk_iters):
                for client_idx in candidates:
                    algclass.client_train(
                        client_idx, train_loaders[client_idx], a_iter)
            # 新代码
            # for client_idx in range(args.n_clients):
            #     for wi in range(args.wk_iters):
            #         algclass.client_train(
            #             client_idx, train_loaders[client_idx], a_iter)

            # server aggregation

        algclass.server_aggre()

        best_acc, best_tacc, best_changed, train_average_loss_history, train_average_acc_history, val_average_loss_history, val_average_acc_history, test_average_acc_history = evalandprint(
            args, algclass, train_loaders, val_loaders, test_loaders, SAVE_PATH, best_acc, best_tacc, a_iter,
            best_changed,
            train_average_loss_history,
            train_average_acc_history,
            val_average_loss_history,
            val_average_acc_history,
            test_average_acc_history)  # 用于评估模型性能并记录最佳性能的变化

    if args.alg == 'metafed':
        print('Personalization stage')
        for c_idx in range(args.n_clients):
            algclass.personalization(
                c_idx, train_loaders[algclass.csort[c_idx]], val_loaders[algclass.csort[c_idx]])
        best_acc, best_tacc, best_changed, train_average_loss_history, train_average_acc_history, val_average_loss_history, val_average_acc_history, test_average_acc_history = evalandprint(
            args, algclass, train_loaders, val_loaders, test_loaders, SAVE_PATH, best_acc, best_tacc, a_iter,
            best_changed,
            train_average_loss_history,
            train_average_acc_history,
            val_average_loss_history,
            val_average_acc_history,
            test_average_acc_history
        )

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
        # './result_test_acc_zhengshibijiao_01fedavg_150_0.01_120/'最后的三个数字是_轮次_学习率_本地训练轮次
        # root_result = './result_fedavg_FVN_60/'
        # root_result = './result_train_data_fedavg_600/'
        # root_result = './huadongchuangk_fedavg_600/'
        # root_result ='./metafed_0.0005_600/'
        # root_result = './FVN_0.0005_600/'
        # root_result = './result_fedbn_600/'
        # root_result = './ceshi_timit_600/'
        # root_result = './ceshi_voxceleb1_600/'
        # root_result = './ceshi_zhvoice_600—xin/'
        root_result ='./zhthchs30_fedbn/'
        os.makedirs(root_result, exist_ok=True)
        os.makedirs(root_result + args.dataset, exist_ok=True)
        file = root_result + args.dataset + '/' + str(args.n_clients)
        os.makedirs(file, exist_ok=True)
        file = file + '/' + str(args.non_iid_alpha) + '.json'
        # 检查并删除旧文件
        if os.path.exists(file):
            os.remove(file)
            print(f'Removing old file: {file}')
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