from alg.fedavg import fedavg
from alg.fedprox import fedprox
from alg.fedbn import fedbn
from alg.base import base
from alg.fedap import fedap
from alg.metafed import metafed
from alg.localmoml import localmoml

ALGORITHMS = [
    'fedavg',
    'fedprox',
    'fedbn',
    'base',
    'fedap',
    'metafed',
    'localmoml'
]


def get_algorithm_class(algorithm_name):
    """返回具有给定名称的算法类。"""
    if algorithm_name not in globals():
        raise NotImplementedError(
            "Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
