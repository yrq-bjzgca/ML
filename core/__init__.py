# core/__init__.py

# ① 核心对象
from .tensor import Tensor        # 可微张量
from . import functional          # 算子命名空间（relu, conv2d...）
from . import optim               # 优化器命名空间（SGD, Adam...）

# ② 控制“from core import *”时的白名单
__all__ = ['Tensor', 'functional', 'optim']