"""
神经网络模型
提供神经网络模型基类和容器
"""

from typing import List, Dict, Iterator

# from ..core.tensor import Tensor
# 在当前文件下调用tensor
import sys
sys.path.append("..")
from core import Tensor
from .layer import Linear, Dropout, BatchNorm1d, BatchNorm2d

class Module:
    """
    神经网络模块基类
    所有神经网络模块都应该继承这个类
    """
    
    def __init__(self):
        """初始化模块"""
        self._modules = {}
        self._parameters = {}
        self.training = True  # 默认处于训练模式
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量
            
        返回:
            输出张量
        """
        raise NotImplementedError("子类必须实现forward方法")
    
    def __call__(self, x: Tensor) -> Tensor:
        """使实例可调用"""
        return self.forward(x)
    
    def parameters(self) -> Iterator[Tensor]:
        """
        返回模块的所有参数
        
        返回:
            参数迭代器
        """
        # TODO: 返回所有可训练参数
        # 包括当前模块的参数和所有子模块的参数
        pass
    
    def add_module(self, name: str, module: 'Module') -> None:
        """
        添加子模块
        
        参数:
            name: 子模块名称
            module: 子模块实例
        """
        # TODO: 添加子模块到_modules字典
        pass
    
    def children(self) -> Iterator['Module']:
        """
        返回直接子模块迭代器
        
        返回:
            子模块迭代器
        """
        # TODO: 返回直接子模块迭代器
        pass
    
    def modules(self) -> Iterator['Module']:
        """
        返回所有模块的迭代器（包括自身）
        
        返回:
            模块迭代器
        """
        # TODO: 返回所有模块的迭代器
        pass
    
    def train(self) -> None:
        """设置为训练模式"""
        # TODO: 设置当前模块和所有子模块为训练模式
        pass
    
    def eval(self) -> None:
        """设置为评估模式"""
        # TODO: 设置当前模块和所有子模块为评估模式
        pass
    
    def zero_grad(self) -> None:
        """清零所有参数的梯度"""
        # TODO: 遍历所有参数，将梯度置零
        pass
    
    def __setattr__(self, name: str, value) -> None:
        """
        设置属性
        
        参数:
            name: 属性名
            value: 属性值
        """
        # TODO: 特殊处理模块和参数的设置
        # 如果value是Module实例，添加到_modules
        # 如果value是Tensor且requires_grad=True，添加到_parameters
        # 否则正常设置属性
        pass
    
    def __getattr__(self, name: str):
        """
        获取属性
        
        参数:
            name: 属性名
            
        返回:
            属性值
        """
        # TODO: 特殊处理模块和参数的获取
        # 如果在_modules中，返回对应模块
        # 如果在_parameters中，返回对应参数
        # 否则抛出AttributeError
        pass


class Sequential(Module):
    """
    顺序容器
    按顺序执行包含的模块
    """
    
    def __init__(self, *args: Module):
        """
        初始化顺序容器
        
        参数:
            *args: 按顺序排列的模块
        """
        super().__init__()
        # TODO: 将所有传入的模块添加到容器中
        pass
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量
            
        返回:
            输出张量
        """
        # TODO: 按顺序执行所有模块的前向传播
        pass
    
    def append(self, module: Module) -> 'Sequential':
        """
        添加模块到容器末尾
        
        参数:
            module: 要添加的模块
            
        返回:
            self，用于链式调用
        """
        # TODO: 添加模块到容器末尾
        pass