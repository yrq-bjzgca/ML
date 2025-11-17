"""
神经网络模型
提供神经网络模型基类和容器
"""

from typing import List, Dict, Iterator, Union

# from ..core.tensor import Tensor
# 在当前文件下调用tensor
import sys
sys.path.append("..")
from core import Tensor
from .layer import Linear, Dropout, BatchNorm1d, BatchNorm2d
import pdb
from .base import Module
import numpy as np

class Sequential(Module):
    """
    顺序容器
    按顺序执行包含的模块
    """
    
    def __init__(self, *modules: Module):
        """
        初始化顺序容器
        
        参数:
            *args: 按顺序排列的模块
        """
        super().__init__()
        # TODO: 将所有传入的模块添加到容器中
        for i, module in enumerate(modules):
            self.add_module(str(i), module)
            
    @property
    def layers(self):
        return self._modules
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量
            
        返回:
            输出张量
        """
        # TODO: 按顺序执行所有模块的前向传播
        for module in self._modules.values():
            x = module(x)
        return x
    
    def append(self, module: Module) -> 'Sequential':
        """
        添加模块到容器末尾
        
        参数:
            module: 要添加的模块
            
        返回:
            self，用于链式调用
        """
        # TODO: 添加模块到容器末尾
        # 获取下一个索引
        next_idx = len(self._modules)
        self.add_module(str(next_idx), module)
        return self

    def __getitem__(self,idx: Union[int, slice])->Module:
        """
        获取指定索引的模块
        
        参数:
            idx: 索引或切片
            
        返回:
            模块或新的Sequential实例
        """  
        if isinstance(idx, slice):
            # 处理切片
            modules = list(self._modules.values())[idx]
            return Sequential(*modules)
        else:
            # 处理单个索引
            return list(self._modules.values())[idx]
        
    def state_dict(self, prefix:str = '')->Dict:
        """
        重写 state_dict 方法，确保平铺结构
        """
        state_dict = {}
        _modules = object.__getattribute__(self, '_modules')
        # 直接平铺所有的子模块的参数
        for name, module in _modules.items():
            full_prefix = f"{prefix}.{name}" if prefix else name
            model_state = module.state_dict(full_prefix)
            state_dict.update(model_state)
        return state_dict
    
    def __len__(self) -> int:
        """返回容器中的模块数量"""
        return len(self._modules)
    
    def __iter__(self) -> Iterator[Module]:
        """返回模块迭代器"""
        return iter(self._modules.values())
    
    def __repr__(self) -> str:
        """返回Sequential的字符串表示"""
        modules_str = ', \n'.join([repr(module) for module in self._modules.values()])
        return f"Sequential({modules_str})"
    
