from .tensor import Tensor
import numpy as np

class GD:...

class SGD:
    """
    随机梯度下降优化器
    
    参数:
        params: 需要优化的参数列表 [Tensor]
        lr: 学习率 (float)
        momentum: 动量系数，0表示不使用动量 (float)
    """
    def __init__(self, params: list[Tensor], lr: float, momentum: float = 0.0):
        self.params = params        # 需要优化的参数
        self.lr = lr                # 学习率
        self.momentum = momentum    # 动量系数
        self.velocities = []        # 动量速度缓存
        # 为每个参数进行缓存
        for param in self.params:
            self.velocities.append(np.zeros_like(param.data))

    def step(self) -> None:
        """
        执行一步参数更新
        更新公式: 
            v = momentum * v - lr * grad
            param += v
        """
        for i, param in enumerate(self.params):
            if param.grad is not None:
                # 计算动量项
                self.velocities[i] = self.momentum * self.velocities[i] - self.lr * param.grad
                # 更新参数
                param.data += self.velocities[i]
    
    #清空所有的参数缓存
    def zero_grad(self) -> None: 
        for param in self.params:
            if param.grad is not None:
                param.zero_grad()

class Momentum:
    """
    动量优化器
    
    参数:
        params: 需要优化的参数列表 [Tensor]
        lr: 学习率 (float)
        momentum: 动量系数，通常设为0.9 (float)
    """
    def __init__(self, params: list[Tensor], lr: float, momentum: float = 0.9):
        self.params = params        # 需要优化的参数
        self.lr = lr                # 学习率
        self.momentum = momentum    # 动量系数
        self.velocities = []        # 动量速度缓存
        # 为每个参数进行缓存
        for param in self.params:
            self.velocities.append(np.zeros_like(param.data))
            
    def step(self) -> None: 
        """
        执行一步参数更新
        更新公式:
            v = momentum * v - lr * grad
            param += v
            
        与SGD的区别：通常使用更大的动量系数
        """
        for i, param in enumerate(self.params):
            if param.grad is not None:
                # 累积动量
                self.velocities[i] = self.momentum * self.velocities[i] - self.lr * param.grad
                param.data += self.velocities[i]
    def zero_grad(self) -> None: 
        for param in self.params:
            if param.grad is not None:
                param.zero_grad()

class AdaGrad:
    """
    AdaGrad优化器 - 自适应学习率
    
    参数:
        params: 需要优化的参数列表 [Tensor]
        lr: 学习率 (float)，通常设为0.01
        eps: 数值稳定性常数，防止除零 (float)
    """
    def __init__(self, params: list[Tensor], lr: float, eps: float = 1e-8): 
        self.params = params        # 需要优化的参数
        self.lr = lr                # 学习率
        self.eps = eps              # 防止除0的小常数
        self.cache = []             # 梯度平方累积缓存
        # 为每个参数进行缓存
        for param in self.params:
            self.cache.append(np.zeros_like(param.data))
           
    def step(self) -> None: 
        """
        执行一步参数更新
        更新公式:
            cache += grad^2
            param -= lr * grad / (sqrt(cache) + eps)
            
        特点: 为每个参数自适应调整学习率，频繁更新的参数学习率较小
        """
        for i, param in enumerate(self.params):
            if param.grad is not None:
                # 累积梯度平方
                self.cache[i] += param.grad ** 2
                # 计算自适应学习率更新
                adaptive_lr = self.lr / (np.sqrt(self.cache[i]) + self.eps)
                param.data -= adaptive_lr * param.grad

    def zero_grad(self) -> None: 
        for param in self.params:
            if param.grad is not None:
                param.zero_grad()

class Adam:
    def __init__(self, params: list[Tensor], lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8): ...
    def step(self) -> None: ...
    def zero_grad(self) -> None: 
        for param in self.params:
            if param.grad is not None:
                param.zero_grad()