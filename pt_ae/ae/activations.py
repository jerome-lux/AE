from typing import Optional
from functools import partial
from torch import nn
from .utils import IdentityLayer


ACT_DICT: dict[str, type] = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
    "silu": nn.SiLU,
    "gelu": partial(nn.GELU, approximate="tanh"),
    "leakyrelu": nn.LeakyReLU,
}


def build_activation(name: str, **kwargs) -> Optional[nn.Module]:

    if name is None:
        return IdentityLayer()
    if name in ACT_DICT:
        act_cls = ACT_DICT[name]
        return act_cls(**kwargs)
    else:
        return IdentityLayer()
