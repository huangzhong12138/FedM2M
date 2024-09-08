import functools
import numpy as np
import math
import os
import torch.distributed as dist
import torch

class Partition(object):
    """ 类似于数据集的对象，但只访问其中的一个子集。 """

    def __init__(self, data, indices):
        self.data = data
        self.indices = indices
        self.replaced_targets = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        data_idx = self.indices[index]
        return (self.data[data_idx][0], self.data[data_idx][1])

    def update_replaced_targets(self, replaced_targets):
        self.replaced_targets = replaced_targets

        # evaluate the the difference between original labels and the simulated labels.
        count = 0
        for index in range(len(replaced_targets)):
            data_idx = self.indices[index]

            if self.replaced_targets[index] == self.data[data_idx][1]:
                count += 1
        return count / len(replaced_targets)

    def set_targets(self, replaced_targets):
        self.replaced_targets = replaced_targets

    def get_targets(self):
        return self.replaced_targets

    def clean_replaced_targets(self):
        self.replaced_targets = None

class DataPartitioner(object):
    """ 将数据集划分为不同的chunck。 """
    def __init__(
        self, conf, data, partition_sizes, partition_type = 'evenly', consistent_indices=True
    ):
        # prepare info.
        self.conf = conf
        self.partition_sizes = partition_sizes
        self.partition_type = partition_type
        self.consistent_indices = consistent_indices
        self.partitions = []

        # get data, data_size, indices of the data.
        # 获取数据，data_size，数据的索引。
        self.data_size = len(data)

        # 检查数据是否已经被分区
        if type(data) is not Partition:
            self.data = data
            indices = np.array([x for x in range(0, self.data_size)])
        else:
            self.data = data.data
            indices = data.indices
        self.partition_indices(indices)


    def partition_indices(self, indices):
        indices = self._create_indices(indices)
        if self.consistent_indices:
            # 这是一个条件判断，检查是否要保持一致的索引。
            indices = self._get_consistent_indices(indices)

        if self.partition_type == 'evenly':
            # 条件判断，检查分区类型是否为 "evenly"，即均匀分区。
            classes = np.unique(self.data.targets)  # 获取原始数据中的唯一类别（目标），存储在 classes 变量。
            lp = len(self.partition_sizes)  # 客户端数量
            ti = indices[:, 0]
            ttar = indices[:, 1]
            for i in range(lp):
                self.partitions.append(np.array([]))
            for c in classes:
                tindice = np.where(ttar == c)[0]
                lti = len(tindice)  # 每个类的个数，类别的数据数量
                from_index = 0
                for i in range(lp):
                    partition_size = self.partition_sizes[i] # 获取当前数据块的分区大小。
                    to_index = from_index + int(partition_size * lti) # 计算数据块的结束索引，确保它不超过当前类别的索引范围。
                    if i == (lp-1):
                        # 根据分区的大小，将数据索引划分到当前数据块中。如果是最后一个数据块，则包含所有剩余的索引。
                        self.partitions[i] = np.hstack(
                            (self.partitions[i], ti[tindice[from_index:]]))
                    else:
                        self.partitions[i] = np.hstack(
                            (self.partitions[i], ti[tindice[from_index:to_index]]))
                    from_index = to_index
            for i in range(lp):
                # 循环将每个数据块的索引从 NumPy 数组转换为整数列表。
                self.partitions[i] = self.partitions[i].astype(np.int).tolist()
        else:
            from_index = 0
            for partition_size in self.partition_sizes:
                to_index = from_index + int(partition_size * self.data_size)
                self.partitions.append(indices[from_index:to_index])
                from_index = to_index

        record_class_distribution(
            self.partitions, self.data.targets
        )


    def _create_indices(self, indices):
        # 该方法用于根据指定的分区类型创建或转换数据索引。
        if self.partition_type == "origin":
            pass
        elif self.partition_type == "random":
            # 将随机打乱索引。
            self.conf.random_state.shuffle(indices)
        elif self.partition_type == 'evenly':
            indices = np.array([
                (idx, target)
                for idx, target in enumerate(self.data.targets)
                if idx in indices
            ])
        elif self.partition_type == "sorted":
            # it will sort the indices based on the data label.
            # 它将根据数据标签对索引进行排序。
            indices = [
                i[0]
                for i in sorted(
                    [
                        (idx, target)
                        for idx, target in enumerate(self.data.targets)
                        if idx in indices
                    ],
                    key=lambda x: x[1],
                )
            ]
        elif self.partition_type == "non_iid_dirichlet":
            num_classes = len(np.unique(self.data.targets))
            num_indices = len(indices)
            n_workers = len(self.partition_sizes)

            list_of_indices = build_non_iid_by_dirichlet(
                random_state=self.conf.random_state,  # 随机数生成器的状态，用于生成随机数。
                indices2targets=np.array(
                    [
                        (idx, target)
                        for idx, target in enumerate(self.data.targets)
                        if idx in indices
                    ]
                ),  # 包含索引和对应目标的NumPy数组，由原始数据集中的索引和目标组成。
                non_iid_alpha=self.conf.non_iid_alpha, # 用于狄利克雷分布的参数，控制非独立同分布数据的生成。
                num_classes=num_classes,
                num_indices=num_indices,
                n_workers=n_workers,
            )
            indices = functools.reduce(lambda a, b: a + b, list_of_indices)
            print("-------------------非独立同分布分区后------------------")
            print("indices的值是：{}".format(indices))
        else:
            raise NotImplementedError(
                f"The partition scheme={self.partition_type} is not implemented yet"
            )
        # build_non_iid_by_dirichlet函数返回一个索引列表的列表，其中每个列表代表一个分区。
        # 原始的索引列表经过狄利克雷分布处理后被重新分配到不同的分区中。
        return indices

    def use(self, partition_ind):
        # 该方法的作用是根据指定的分区索引 partition_ind 返回一个新的 Partition 对象，
        # 该对象包含了特定分区的数据索引。
        return Partition(self.data, self.partitions[partition_ind])


    def _get_consistent_indices(self, indices):
        if dist.is_initialized():
            # sync the indices over clients.
            # 通过客户端同步索引。
            indices = torch.IntTensor(indices)
            dist.broadcast(indices, src=0)
            return list(indices)
        else:
            return indices







def getdataloader(conf, dataall, root='./split/'):
    print("待会在写")
    file = root + "ceshi" + ".npy"
    if not os.path.exists(file):
        data_part = define_data_loader(conf, dataall)  # 用于定义数据加载器并准备数据。
        tmparr = []  # 存储数据分区
        for i in range(conf.n_clients):
            tmppart = define_val_dataset(conf, data_part.use(i))
            tmparr.append(tmppart.partitions[0])
            tmparr.append(tmppart.partitions[1])
            tmparr.append(tmppart.partitions[2])
        tmparr = np.array(tmparr, dtype=object)
        np.save(file, tmparr)
    else:
        conf.partition_data = 'origin'
        data_part = define_data_loader(conf, dataall)
    data_part.partitions = np.load(file, allow_pickle=True).tolist()
    clienttrain_list = []
    clientvalid_list = []
    clienttest_list = []
    for i in range(conf.n_clients):
        clienttrain_list.append(data_part.use(3 * i))
        clienttest_list.append(data_part.use(3 * i + 1))
        clientvalid_list.append(data_part.use(3 * i + 2))
    return clienttrain_list, clientvalid_list, clienttest_list



def define_val_dataset(conf, train_dataset):
    # 基于给定的训练数据集创建一个验证数据集的分区对象
    # 数据集分区
    partition_sizes = [
        0.6, 0.1, 0.3
    ]
    # partition_sizes = [
    #     0.8, 0, 0.2
    # ]
    # 训练数据集分割
    data_partitioner = DataPartitioner(
        conf,
        train_dataset,
        partition_sizes,
        conf.partition_data
        # partition_type="evenly",
        # consistent_indices=False,
    )
    return data_partitioner


def define_data_loader(conf, dataset, data_partitioner=None):
    world_size = conf.n_clients
    partition_sizes = [1.0 / world_size for _ in range(world_size)]
    # partition_sizes = [0.2, 0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1]
    print(partition_sizes)
    if data_partitioner is None:
        # 更新数据分区器。
        print(None)
        data_partitioner = DataPartitioner(
            conf, dataset, partition_sizes, partition_type=conf.partition_data
        )
    return data_partitioner

# 这个暂时不看，应该是non-iid
def build_non_iid_by_dirichlet(
        random_state, indices2targets, non_iid_alpha, num_classes, num_indices, n_workers
):
    """
    refer to https://github.com/epfml/quasi-global-momentum/blob/3603211501e376d4a25fb2d427c30647065de8c8/code/pcode/datasets/partition_data.py
    """
    n_auxi_workers = 2
    assert n_auxi_workers <= n_workers

    # random shuffle targets indices.
    random_state.shuffle(indices2targets)

    # partition indices.
    from_index = 0
    splitted_targets = []
    num_splits = math.ceil(n_workers / n_auxi_workers)
    split_n_workers = [
        n_auxi_workers
        if idx < num_splits - 1
        else n_workers - n_auxi_workers * (num_splits - 1)
        for idx in range(num_splits)
    ]
    split_ratios = [_n_workers / n_workers for _n_workers in split_n_workers]
    for idx, _ in enumerate(split_ratios):
        to_index = from_index + int(n_auxi_workers / n_workers * num_indices)
        splitted_targets.append(
            indices2targets[
            from_index: (num_indices if idx ==
                                        num_splits - 1 else to_index)
            ]
        )
        from_index = to_index

    idx_batch = []
    for _targets in splitted_targets:
        # rebuild _targets.
        _targets = np.array(_targets)
        _targets_size = len(_targets)

        # use auxi_workers for this subset targets.
        _n_workers = min(n_auxi_workers, n_workers)
        n_workers = n_workers - n_auxi_workers

        # get the corresponding idx_batch.
        min_size = 0
        while min_size < int(0.50 * _targets_size / _n_workers):
            _idx_batch = [[] for _ in range(_n_workers)]
            for _class in range(num_classes):
                # get the corresponding indices in the original 'targets' list.
                idx_class = np.where(_targets[:, 1] == _class)[0]
                idx_class = _targets[idx_class, 0]

                # sampling.
                try:
                    proportions = random_state.dirichlet(
                        np.repeat(non_iid_alpha, _n_workers)
                    )
                    # balance
                    proportions = np.array(
                        [
                            p * (len(idx_j) < _targets_size / _n_workers)
                            for p, idx_j in zip(proportions, _idx_batch)
                        ]
                    )
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[
                                  :-1
                                  ]
                    _idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(
                            _idx_batch, np.split(idx_class, proportions)
                        )
                    ]
                    sizes = [len(idx_j) for idx_j in _idx_batch]
                    min_size = min([_size for _size in sizes])
                except ZeroDivisionError:
                    pass
        idx_batch += _idx_batch
    return idx_batch


def record_class_distribution(partitions, targets):
    # 记录不同数据分区中各类别的分布信息
    targets_of_partitions = {}
    targets_np = np.array(targets) # 将目标标签列表 targets 转换为 NumPy 数组，以便进行数组操作。
    for idx, partition in enumerate(partitions):
        unique_elements, counts_elements = np.unique(
            targets_np[partition], return_counts=True
        )
        targets_of_partitions[idx] = list(
            zip(unique_elements, counts_elements))
    return targets_of_partitions





#下面这些方法还不会
def define_pretrain_dataset(conf, train_dataset):
    partition_sizes = [
        0.3, 0.7
    ]
    data_partitioner = DataPartitioner(
        conf,
        train_dataset,
        partition_sizes,
        partition_type="evenly",
        # consistent_indices=False,
    )
    return data_partitioner.use(0)




# if __name__ == '__main__':
    # list =[]
    # define_data_loader(dataset=list)
