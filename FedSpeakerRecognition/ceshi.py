import numpy as np
import torch


class huangzhong():
    def __init__(self):
        print("huangzhong")

    def __len__(self):
        print(1234566)

    def dongzuo(self):
        i = 1000
        print(10000)


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')



if __name__ == '__main__':
    import torch
    import time
    from torch import autograd

    # GPU加速
    print(torch.__version__)
    print(torch.cuda.is_available())

    a = torch.randn(10000, 1000)
    b = torch.randn(1000, 10000)
    print(a)
    print(b)
    t0 = time.time()
    c = torch.matmul(a, b)
    t1 = time.time()

    print(a.device, t1 - t0, c.norm(2))

    device = torch.device('cuda')
    print(device)
    a = a.to(device)
    b = b.to(device)

    t0 = time.time()
    c = torch.matmul(a, b)
    t2 = time.time()
    print(a.device, t2 - t0, c.norm(2))

    t0 = time.time()
    c = torch.matmul(a, b)
    t2 = time.time()

    print(a.device, t2 - t0, c.norm(2))


