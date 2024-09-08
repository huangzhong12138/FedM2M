import enum
import numpy as np
import torch


def evalandprint(args, algclass, train_loaders, val_loaders, test_loaders, SAVE_PATH, best_acc, best_tacc, a_iter,
                 best_changed,
                 train_average_loss_history, train_average_acc_history, val_average_loss_history,
                 val_average_acc_history, test_average_acc_history):
    # evaluation on training data
    test_acc_list = []
    train_loss_list = []
    train_acc_list = []
    for client_idx in range(args.n_clients):
        train_loss, train_acc = algclass.client_eval(
            client_idx, train_loaders[client_idx])
        print(
            f' Site-{client_idx:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        # 保存一系列值，准备进行平均
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

    # evaluation on valid data
    val_loss_list = []

    val_acc_list = [None] * args.n_clients  # 这个值就是保存每个客户端的准确度，可以直接拿来用
    for client_idx in range(args.n_clients):
        val_loss, val_acc = algclass.client_eval(
            client_idx, val_loaders[client_idx])
        val_acc_list[client_idx] = val_acc
        print(
            f' Site-{client_idx:02d} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

        # 保存一系列值，准备进行平均
        val_loss_list.append(val_loss)

    # 联邦平均才注解，因为不想要best_changed
    if np.mean(val_acc_list) > np.mean(best_acc):
        for client_idx in range(args.n_clients):
            best_acc[client_idx] = val_acc_list[client_idx]
            best_epoch = a_iter
        best_changed = True
    # if args.alg == 'fedavg' or args.alg == 'metafed' or args.alg =='fedbn':
    # 获取联邦平均每一次的test_acc
    for client_idx in range(args.n_clients):
        _, test_acc = algclass.client_eval(
            client_idx, test_loaders[client_idx])
        test_acc_list.append(test_acc)

    if best_changed:
        best_changed = False
        # test
        for client_idx in range(args.n_clients):
            _, test_acc = algclass.client_eval(
                client_idx, test_loaders[client_idx])
            print(
                f' Test site-{client_idx:02d} | Epoch:{best_epoch} | Test Acc: {test_acc:.4f}')
            best_tacc[client_idx] = test_acc
        print(f' Saving the local and server checkpoint to {SAVE_PATH}')
        tosave = {'best_epoch': best_epoch, 'best_acc': best_acc, 'best_tacc': np.mean(np.array(best_tacc))}
        for i, tmodel in enumerate(algclass.client_model):
            tosave['client_model_' + str(i)] = tmodel.state_dict()
        tosave['server_model'] = algclass.server_model.state_dict()
        torch.save(tosave, SAVE_PATH)

    train_average_loss_history.append(sum(train_loss_list) / len(train_loss_list))
    train_average_acc_history.append(sum(train_acc_list) / len(train_acc_list))
    val_average_loss_history.append(sum(val_loss_list) / len(val_loss_list))
    val_average_acc_history.append(sum(val_acc_list) / len(val_acc_list))
    if len(test_acc_list) > 0:
        test_average_acc_history.append(sum(test_acc_list) / len(test_acc_list))

    return best_acc, best_tacc, best_changed, train_average_loss_history, train_average_acc_history, val_average_loss_history, val_average_acc_history, test_average_acc_history
