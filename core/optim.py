from .tensor import Tensor
import numpy as np
from typing import List, Union
import warnings

# 当前文件测试的时候取消下面注释
# from tensor import Tensor
class Optimizer:
    """
    优化器的基类
    提供参数管理，梯度处理和数组稳定性检查
    """
    def __init__(self, params: List, lr: float = 1e-3):
        """
        初始化优化器
        
        参数:
            params: 需要优化的参数列表
            lr: 学习率
        """
        # self.params = params
        self.params = list(params)
        self.lr = lr
        
        # 鲁棒性检查
        self._validate_params()
        
        # 初始化状态字典（子类可以扩展）
        self.state = {}

    def _validate_params(self) -> None:
        """验证参数列表"""
        if not self.params:
            raise ValueError("param is empty, by yrq")
        
        for i, param in enumerate(self.params):
            if not hasattr(param, 'data'):
                raise ValueError(f"param {i} don't have attribute data, by yrq")
            if not hasattr(param, 'grad'):
                raise ValueError(f"param {i} don't have attribute grad, by yrq")
            
    # 安全将梯度广播到目标
    def _safe_broadcast_grad(self, grad:np.ndarray, target_shape:tuple)->np.ndarray:
        # 如果形状相同，直接返回
        if grad.shape == target_shape:
            return grad
        
        #尝试进行广播
        try:
            #直接广播
            broad_grad = np.broadcast_to(grad, target_shape)
            return broad_grad
        except ValueError:
            # 如果广播失败，尝试求和到目标的形状
            # 找到需要的求和的轴
            axes = self._get_sum_axis(grad, target_shape)
            if axes:
                summed_grad = grad.sum(axis = axes, keepdims = True)
                #再次尝试广播
                return np.broadcast_to(summed_grad, target_shape)
            else:
                raise ValueError(f"can't broadcast from {grad.shape} to {target_shape}, by yrq")
                
    def _get_sum_axis(self, grad_shape:tuple, target_shape:tuple)->tuple:
        """
        计算求和需要的轴
        参数：
            grad_shape:梯度形状
            target_shape:目标形状

        广播规则：从右向左比较维度
        - 如果维度相等，或其中一个为1，可以广播
        - 如果梯度维度>1且目标维度=1，需要在这个轴求和
        - 如果梯度维度=1且目标维度>1，可以直接广播

        返回：
            需要求和的轴
        """

        # 确保输入是元组
        if not isinstance(grad_shape, tuple):
            grad_shape = grad_shape.shape if hasattr(grad_shape, 'shape') else tuple(grad_shape)
        if not isinstance(target_shape, tuple):
            target_shape = target_shape.shape if hasattr(target_shape, 'shape') else tuple(target_shape)
        
        grad_ndim = len(grad_shape)
        target_ndim = len(target_shape)

        # 如果梯度维度更多，前几个维度需要求和
        if grad_ndim>target_ndim:
            return tuple(range(grad_ndim - target_ndim))
        
        # 否则找到需要求和的轴，梯度为1但是目标中没有1的轴
        axes = []

        min_ndim = min(target_ndim, grad_ndim)

        # 从右边开始广播
        for i in range(1, min_ndim + 1):
            grad_dim = grad_shape[-i]
            target_dim = target_shape[-i]

            if grad_dim != 1 and target_dim ==1:
                axes.append(grad_ndim - i)

        return tuple(axes)
    
    def _check_numerical_stability(self, array: np.ndarray, name: str) -> bool:
        """
        检查数值稳定性
        
        参数:
            array: 要检查的数组
            name: 数组名称（用于错误信息）
            
        返回:
            True如果稳定，False如果不稳定
        """
        if np.any(np.isnan(array)):
            warnings.warn("{name} contain nan, by yrq")
            return False
        if np.any(np.isinf(array)):
            warnings.warn("{name} contain inf, by yrq")
            return False
        return True
    
    def _clip_if_large(self, array: np.ndarray, threshold: float = 1e3) -> np.ndarray:
        """
        如果数组值过大则进行裁剪
        
        参数:
            array: 要检查的数组
            threshold: 阈值
            
        返回:
            裁剪后的数组
        """
        if np.max(np.abs(array)) > threshold:
            warnings.warn("the aray size is large need clip, by yrq")
            return np.clip(array, -threshold, threshold)
        return array

    def step(self)->None:
        """
        执行一步参数更新
        """
        raise NotImplementedError("子类必须实现step的方法")
    

    #清空所有的参数缓存
    def zero_grad(self) -> None: 
        for param in self.params:
            # if param.grad is not None:
            #     param.zero_grad()
            if hasattr(param, 'zero_grad'):
                param.zero_grad()
            elif param.grad is not None:
                param.grad.fill(0.0)

class GD:...


class SGD(Optimizer):
    def __init__(self, params:List, lr: float = 1e-3, momentum: float = 0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        # 初始化速度缓存
        for i,param in enumerate(self.params):
            self.state[f'velocity_{i}'] = np.zeros_like(param.data)

    def step(self):
        for i,param in enumerate(self.params):
            if param.grad is None:
                continue
            if param.grad.shape != param.data.shape:
                grad = self._safe_broadcast_grad(param.grad, param.data.shape)
            else:
                grad = param.grad
            
            if not self._check_numerical_stability(grad, f"the param {i} grad"):
                continue
            

            # 获取速度
            velocity = self.state[f'velocity_{i}']
            
            # 更新动量
            velocity = self.momentum * velocity - self.lr * grad
            velocity = self._clip_if_large(velocity)

            # 保存更新后的速度
            self.state[f'velocity_{i}'] = velocity
            
            # 更新参数
            param.data += velocity
            
            # 检查参数稳定性
            self._check_numerical_stability(param.data, f"param {i}")
    
class Momentum(SGD):
    """
    动量优化器
    
    参数:
        params: 需要优化的参数列表 [Tensor]
        lr: 学习率 (float)
        momentum: 动量系数，通常设为0.9 (float)
    """
    def __init__(self,  params: list[Tensor], lr: float, momentum: float = 0.9):
        super().__init__(params, lr, momentum)

class AdaGrad(Optimizer):
    """
    AdaGrad优化器 - 自适应学习率
    
    参数:
        params: 需要优化的参数列表 [Tensor]
        lr: 学习率 (float)，通常设为0.01
        eps: 数值稳定性常数，防止除零 (float)
    """
    def __init__(self, params: list[Tensor], lr: float, eps: float = 1e-8): 
        super().__init__(params, lr)
        self.eps = eps
        # 初始化速度缓存
        for i,param in enumerate(self.params):
            self.state[f'cache_{i}'] = np.zeros_like(param.data)

    def step(self):
        for i,param in enumerate(self.params):

            if param.grad is None:
                continue
            if param.grad.shape != param.data.shape:
                grad = self._safe_broadcast_grad(param.grad, param.data.shape)
            else:
                grad = param.grad
            
            if not self._check_numerical_stability(grad, f"the param {i} grad"):
                continue

            # 获取cache
            cache = self.state[f'cache_{i}']
            
            # 更新缓存
            cache += grad**2
            cache = self._clip_if_large(cache)

            # 保存更新之后的缓存
            self.state[f'cache_{i}'] = cache

            #计算自适应学习率
            cache_sqrt = np.sqrt(cache) + self.eps
            cache_sqrt = self._clip_if_large(cache_sqrt)
            
            # 更新参数
            param.data -= self.lr * param.grad/cache_sqrt

            # 检查参数稳定性
            self._check_numerical_stability(param.data, f"param {i}")


class Adam(Optimizer):
    """
    Adam优化器 - 结合动量和自适应学习率
    
    参数:
        params: 需要优化的参数列表 [Tensor]
        lr: 学习率 (float)，通常设为1e-3
        beta1: 一阶矩估计的衰减率 (float)
        beta2: 二阶矩估计的衰减率 (float)  
        eps: 数值稳定性常数 (float)
    """
    
    def __init__(self, params: list, lr: float = 1e-3, \
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        # self.params = params
        # self.lr = lr
        # self.beta1 = beta1  # 一阶矩衰减率
        # self.beta2 = beta2  # 二阶矩衰减率
        # self.eps = eps
        # self.t = 0  # 时间步
        
        # # 初始化矩估计
        # self.m = []  # 一阶矩（类似动量）
        # self.v = []  # 二阶矩（类似AdaGrad的缓存）
        
        # for param in self.params:
        #     self.m.append(np.zeros_like(param.data))
        #     self.v.append(np.zeros_like(param.data))
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0 #时间步

        for i,param in enumerate(self.params):
            self.state[f'm_{i}'] = np.zeros_like(param.data) #一阶矩
            self.state[f'v_{i}'] = np.zeros_like(param.data) #二阶矩

    def step(self) -> None:
        """
        执行一步参数更新
        更新公式:
            t = t + 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad^2
            m_hat = m / (1 - beta1^t)
            v_hat = v / (1 - beta2^t)
            param -= lr * m_hat / (sqrt(v_hat) + eps)
        """
        self.t += 1  # 更新时间步
        

        for i,param in enumerate(self.params):
            if param.grad is None:
                continue
            if param.grad.shape != param.data.shape:
                grad = self._safe_broadcast_grad(param.grad, param.data.shape)
            else:
                grad = param.grad
            
            if not self._check_numerical_stability(grad, f"the param {i} grad"):
                continue

            # 获取m,v
            m = self.state[f'm_{i}']
            v = self.state[f'v_{i}']

            # 更新一阶矩估计（带偏差）
            m = self.beta1 * m + (1 - self.beta1) * grad
            m = self._clip_if_large(m)
            # 更新二阶矩估计（带偏差）
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
            v = self._clip_if_large(v)

            self.state[f'm_{i}'] = m
            self.state[f'v_{i}'] = v

            # 计算偏差修正后的估计
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            
            # 更新参数
            v_hat_sqrt = np.sqrt(v_hat)
            if np.any(v_hat_sqrt<self.eps):
                v_hat_sqrt = np.maximum(v_hat_sqrt,self.eps)
            v_hat_sqrt = self._clip_if_large(v_hat_sqrt)
            param.data -= self.lr * m_hat / v_hat_sqrt

            # 检查参数稳定性
            self._check_numerical_stability(param.data, f"param {i}")
