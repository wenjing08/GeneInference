import torch
from torch.nn.modules import Module

class MyActivation(Module):
    def __init__(self, func=None):
        super(MyActivation, self).__init__()
        self.activation_func = func

    def __call__(self, _input, *args, **kwargs):
        return self.activation_func(_input)


def power_x(k1=1, k2=2):
    """"定义闭包函数
    """

    def func_power_x(_input):

        a = _input[:, 0:500].pow(k1)
        b = _input[:, 500:].pow(k2)
        return torch.cat((a,b), 1)
    return func_power_x


def func_exponent_x(_input):
    return torch.exp(-_input.pow(2))