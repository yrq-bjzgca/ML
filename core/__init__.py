# core/__init__.py

# core/__init__.py  （可选扩展）
from .tensor import Tensor
from .functional import relu, sigmoid, cross_entropy   # 最常用的算子
from . import functional, optim

__all__ = ['Tensor', 'relu', 'sigmoid', 'cross_entropy', 'functional', 'optim']