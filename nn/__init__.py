"""
神经网络模块
提供构建神经网络所需的各种层、模型和初始化方法
"""

# 从 layer 模块导入层类
from .layer import (
    Linear,
    Dropout,
    BatchNorm1d,
    BatchNorm2d
)

# 从 model 模块导入模型类
from .model import (
    Module,
    Sequential
)

# 从 init 模块导入初始化方法
from .init import (
    xavier_uniform_,
    xavier_normal_,
    kaiming_uniform_,
    kaiming_normal_,
    normal_,
    uniform_,
    constant_,
    zeros_,
    ones_
)

# 定义公开的API
__all__ = [
    # 层类
    'Linear',
    'Dropout', 
    'BatchNorm1d',
    'BatchNorm2d',
    
    # 模型类
    'Module',
    'Sequential',
    
    # 初始化方法
    'xavier_uniform_',
    'xavier_normal_', 
    'kaiming_uniform_',
    'kaiming_normal_',
    'normal_',
    'uniform_',
    'constant_',
    'zeros_',
    'ones_'
]

# 版本信息
__version__ = "0.1.0"