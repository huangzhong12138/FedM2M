import os
import glob
import json
from scipy.interpolate import interp1d
import numpy as np

if __name__ == '__main__':
    # data_root = './data_result_aotudl/result_localmoml_random'
    # data_root = './result_fenbu_data_fedavg'
    # data_root = './result_train_data_fedavg_600/'
    # data_root = './zhthchs30/'
    data_root ='./zhthchs30_fedbn/'
    # data_root = './zhstcmds_shengxia_ceshi_XXX/'
    json_files = glob.glob(os.path.join(data_root, "*", "*", "*.json"))
    print(json_files)

    heterogeneity_to_addresses = dict()
    heterogeneity_list = []
    for json_file in json_files:
        basename = os.path.basename(json_file)
        # 截取json的文件名，得到异质性参数
        heterogeneity_name = os.path.splitext(basename)
        heterogeneity = heterogeneity_name[0]

        if heterogeneity not in heterogeneity_to_addresses:
            heterogeneity_to_addresses[heterogeneity] = [json_file.replace('\\', '/')]
            heterogeneity_list.append(heterogeneity)
        else:
            heterogeneity_to_addresses[heterogeneity].append(json_file.replace('\\', '/'))
        # 一般来说不会有重复的

    print(heterogeneity_to_addresses)
    print(heterogeneity_list)

    # 开始画图
    heterogeneity_to_data = dict()
    for i in range(len(heterogeneity_list)):
        # print(heterogeneity_list[i])
        # print(heterogeneity_to_addresses.get(heterogeneity_list[i]))
        if heterogeneity_list[i] not in heterogeneity_to_data:
            with open(heterogeneity_to_addresses.get(heterogeneity_list[i])[0], "r", encoding="utf-8") as f:
                content = json.load(f)
            heterogeneity_to_data[heterogeneity_list[i]] = content

    print(heterogeneity_to_data)

    import matplotlib.pyplot as plt
    import matplotlib
    from scipy.interpolate import make_interp_spline

    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']

    datas = heterogeneity_to_data  # 简单名字
    color_list = ['blue', 'red', 'green', 'black', 'yellow', 'cyan', 'magenta']
    # print(len(datas))
    for num in range(len(datas)):
        # "val_average_acc_history" 拿这个来做y
        name = heterogeneity_list[num]  # 这个值是取每个异质性参数
        number_epoch = len(datas.get(name).get('test_average_acc_history'))  # 求epoch，就是x轴
        x_number = list(range(1, number_epoch + 1))  # 形成x轴集合
        y_number = datas.get(name).get('test_average_acc_history')  # 取y轴
        x_smooth = x_number
        y_smooth = y_number

        plt.plot(x_smooth, y_smooth, label=str("异质性" + name), color=color_list[num], linewidth=0.8)

    plt.legend()
    # 添加标签和标题
    plt.xlabel('训练轮次')
    plt.ylabel('测试准确度')
    plt.title('数据异质性对于模型准确度的影响')

    # 显示图形
    plt.show()

    # # 进行简单测试
    # root_dir = 'result/librispeech/10/1-30.json'
    # heterogeneity_to_data = dict()
    # zhi = 1
    # with open(root_dir, "r", encoding="utf-8") as f:
    #     content = json.load(f)
    #     heterogeneity_to_data[zhi] = content
    # # print(content)
    # # print(content.get('train_average_loss_history'))
    # print(heterogeneity_to_data)
    # print(heterogeneity_to_data.get(1))
    # print(heterogeneity_to_data.get(1).get('train_average_loss_history'))

