import os
import glob
import json
from scipy.interpolate import interp1d
import numpy as np

if __name__ == '__main__':
    data_root = './resultceshiecapatdnn_test_acc/librispeech/20'
    # json_files = glob.glob(os.path.join(data_root, "*", "*", "*.json"))
    json_files = glob.glob(os.path.join(data_root,  "*.json"))
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
        number_epoch = len(datas.get(name).get('val_average_acc_history'))  # 求epoch，就是x轴
        x_number = list(range(1, number_epoch+1))  # 形成x轴集合
        y_number = datas.get(name).get('val_average_acc_history')  # 取y轴
        y_number1 = datas.get(name).get('test_average_acc_history')  # 取y轴
        # 计算 Z 得分以识别异常值
        # z_scores = (y_number - np.mean(y_number)) / np.std(y_number)
        # threshold = 2  # 根据需要调整阈值
        # outlier = np.abs(z_scores) > threshold # 查找异常值

        # 排除异常值
        # x = [x for i, x in enumerate(range(1, number_epoch + 1)) if not outlier[i]]
        # y = [y for i, y in enumerate(y_number) if not outlier[i]]
        # 得出x,y值

        # 使用平滑插值方法生成平滑曲线
        # f = interp1d(x, y, kind='cubic')
        # x_smooth = np.linspace(min(x), max(x), 300)
        # y_smooth = f(x_smooth)
        # y_smooth1 = f(x_smooth)
        # y_smooth = make_interp_spline(x_number, y_number)(x_number)  # 平滑曲线设计

        # spline = make_interp_spline(x, y)
        # x_smooth = np.linspace(min(x), max(x), 1000)
        # y_smooth = spline(x_smooth)

        # plt.plot(x_number, y_number, label=str("异质性"+name), color='blue',linewidth=0.8)
        plt.plot(x_number, y_number1, label=str("异质性" + name), color=color_list[num], linewidth=0.8)

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
