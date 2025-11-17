"""
神经网络层
提供各种神经网络层实现
"""

import numpy as np

# from ..core.tensor import Tensor
# from ..core import functional as F

# 在当前文件下调用tensor
import sys
sys.path.append("..")
from core import Tensor
from core import functional as F


from .init import kaiming_normal_, zeros_, kaiming_uniform_, ones_
from .base import Module
import pdb

class Linear(Module):
    """
    全连接层
    实现 y = xW^T + b 的线性变换
    
    参数:
        in_features: 输入特征数
        out_features: 输出特征数  
        bias: 是否使用偏置项
        device: 设备类型（预留）
        dtype: 数据类型（预留）

    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        初始化全连接层
        
        参数:
            in_features: 输入特征数
            out_features: 输出特征数
            bias: 是否使用偏置项
        """
        # TODO: 初始化权重和偏置参数
        # 使用合适的初始化方法初始化self.weight
        # 如果bias为True，初始化self.bias
        # 注册参数以便优化器可以找到它们

        super().__init__() #必须使用父类进行初始化

        if in_features <= 0:
            raise ValueError(f"in_feature must be positive integer, but is {in_features}")
        if out_features <= 0:
            raise ValueError(f"out_feature must be postive integer, but is {out_features}")
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        # 初始化参数
        self.weight = Tensor(
            np.empty((out_features,in_features),dtype=np.float32),
            requires_grad = True,
            name=f"Linear_weight_{in_features}x{out_features}"
        )
      
        self.register_parameter('weight', self.weight)
        # 初始化偏置参数
        if bias:
            self.bias_param = Tensor(
                np.empty(out_features, dtype=np.float32),
                requires_grad=True,
                name=f"Linear_bias_{out_features}"
            )
     
            self.register_parameter('bias', self.bias_param)
        else:
            self.bias_param =None

        self.reset_parameters()

    def register_parameter(self, name: str, tensor: Tensor) -> None:
        """
        安全地注册参数
        
        参数:
            name: 参数名称
            tensor: 参数张量
        """
        if not isinstance(tensor,Tensor):
            raise TypeError(f"parameter must be Tensor, but get the {type(tensor)}")
        if not tensor.requires_grad:
            raise ValueError("register parameter must need grad")
        # 获取_parameter
        _parameter = object.__getattribute__(self, '_parameters')
        # 注册参数
        _parameter[name] = tensor
        object.__setattr__(self, name, tensor)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 (batch_size, in_features) 或 (in_features,)
            
        返回:
            输出张量，形状为 (batch_size, out_features) 或 (out_features,)
        """
        # TODO: 实现前向传播
        # 计算 x @ self.weight.T + self.bias (如果存在)
        if not isinstance(x, Tensor):
            raise ValueError(f"input must be the tensor, but is {type(x)}")
        # 处理1D的输入
        if x.ndim == 1:
            if x.shape[0]!=self.in_features:
                raise ValueError(
                    f"input is not pair, expect {self.in_features} but get {x.shape[0]}"
                )
            x_2d = x.reshape(1,-1)
            output = x_2d@self.weight.T
            if self.bias_param is not None:
                output = output + self.bias_param
            # 移除批次维度
            return output.reshape(-1)
        # 处理2D输入
        elif x.ndim ==2:
            if x.shape[-1]!=self.in_features:
                raise ValueError(
                    f"input dimension is not pair, expect is {self.in_features}, but get {x.shape[1]}"
                )
            output = x@self.weight.T
            if self.bias_param is not None:
                output = output + self.bias_param
            return output
        
        else:
            raise ValueError(f"input dimension is 1d or 2d, but get the {x.ndim} dimension")

    def __call__(self, x: Tensor) -> Tensor:
        """使实例可调用"""
        return self.forward(x)
    
    def parameters(self):
        """
        返回层的所有参数
        
        返回:
            参数列表
        """
        # TODO: 返回所有可训练参数
        params = [self.weight]
        if self.bias_param is not None:
            params.append(self.bias_param)
        return params

    def extra_repr(self) -> str:
        """
        返回层的额外描述信息，用于__repr__
        """
        return f"in_feature = {self.in_features}, out_feature = {self.out_features}, bias = {self.bias}"
    
    def __repr__(self) -> str:
        return f"Linear({self.extra_repr()})"
    
    def reset_parameters(self)->None:
        """
        重新初始化参数
        """
        kaiming_uniform_(self.weight,a = np.sqrt(5), nonlinearity='relu')

        # 重新初始化偏置
        if self.bias_param is not None:
            zeros_(self.bias_param)

class Dropout(Module):
    """
    Dropout层
    在训练期间随机将部分输入元素置零，防止过拟合
    """
    
    def __init__(self, p: float = 0.5, inplace: bool = False):
        """
        初始化Dropout层
        参数:
            p: 元素被置零的概率，默认 0.5
            inplace: 是否原地操作，默认 False
        """
        # TODO: 初始化参数
        # 设置dropout概率
        # 初始化训练模式标志
        super().__init__() #必须使用父类进行初始化
        if p < 0 or p > 1:
            raise ValueError(f"Dropout possibility must be [0,1], but the value is {p}") 
        self.p = p
        self.inplace = inplace
        self.training = True #默认处于训练模式
        self.mask = None #保存dropout掩码，用于反向传播
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量
            
        返回:
            输出张量
        """
        # TODO: 实现前向传播
        # 如果在训练模式，随机生成mask并应用
        # 如果在评估模式，直接返回输入
        if not isinstance(x, Tensor):
            raise TypeError(f"input must be tensor, but is {type(x)}")
        if not self.training or self.p ==0:
            if self.inplace:
                return x
            else:
                return x.copy()
        # 如果是1，全部丢弃，输出0
        if self.p==1:
            if self.inplace:
                x.data = np.zeros_like(x.data)
                return x
            else:
                return Tensor(np.zeros_like(x.data), requires_grad=True)
        # 在训练模式下使用dropout
        # 生成随机掩码，1表示保留，0表示放弃
        scale = 1.0 /(1.0 - self.p)# 缩放因子，保持期望值不变
        # 生成随机掩码与输入形状相同
        mask_data = np.random.binomial(1,1-self.p,x.shape)
        self.mask = Tensor(mask_data * scale, requires_grad=False)

        # 应用层dropout
        if self.inplace:
            x.data *= self.mask.data
            return x
        else:
            return x*self.mask

    def __call__(self, x: Tensor) -> Tensor:
        """使实例可调用"""
        return self.forward(x)
    
    def train(self):
        """设置为训练模式"""
        # TODO: 设置训练模式
        self.training = True
    
    def eval(self):
        """设置为评估模式"""
        # TODO: 设置评估模式
        self.training = False

    def parameters(self):
        """
        Dropout 没有可以返回的训练参数
        """
        # TODO: 返回所有可训练参数
        return []

    def extra_repr(self) -> str:
        """
        返回层的额外描述信息，用于__repr__
        """
        return f"p={self.p}, inplace={self.inplace}"
    
    def __repr__(self) -> str:
        return f"Dropout({self.extra_repr()})"

class BatchNorm1d(Module):
    """
    一维批归一化层
    对小型批量的数据进行归一化
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, 
                 affine: bool = True, track_running_stats: bool = True):
        """
        初始化批归一化层
        
        参数:
            num_features: 特征数
            eps: 数值稳定性常数，防止除零
            momentum: 运行均值和方差的动量
            affine: 是否学习缩放和偏移参数
            track_running_stats: 是否跟踪运行统计量
        """
        # TODO: 初始化参数
        # 初始化可学习的缩放和偏移参数
        # 初始化运行均值和方差
        # 设置其他超参数
        super().__init__() #必须使用父类进行初始化
        if num_features <=0:
            raise ValueError(f"num_feature must be postive num, but the num is {num_features}")
        if eps<0:
            raise ValueError(f"eps must be the positive number,but get{eps}")
        if momentum<0 or momentum>1:
            raise ValueError(f"momentum must between [0,1], but get {momentum}")
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # 可学习缩放和偏移参数
        if affine:
            self.weight = Tensor(
                np.ones(num_features, dtype=np.float32),
                requires_grad= True,
                name=f"Linear_weight_{num_features}"
            )
            # self.register_parameter('weight', self.weight)
            self.bias = Tensor(
                np.zeros(num_features,dtype=np.float32),
                requires_grad=True,
                name=f"Linear_bias_{num_features}"
            )
            # self.register_parameter('bias', self.bias)
        else:
            self.weight = None
            self.bias = None

       # 运行统计量（用于评估模式）
        if track_running_stats:
            self.running_mean = Tensor(
                np.zeros(num_features,dtype=np.float32),
                requires_grad=False
            )
     
            self.running_var = Tensor(
                np.ones(num_features,dtype=np.float32),
                requires_grad=False
            )
      
        else:
            self.running_mean = None
            self.running_var = None


        
        # 确保正确初始化 current_mean 和 current_var
        object.__setattr__(self, 'current_mean', None)
        object.__setattr__(self, 'current_val', None)
        
        # self.current_mean = None
        # self.current_val = None
        # 评估/训练模式
        self.training = True

        self.reset_parameters()

    def register_parameter(self, name: str, tensor: Tensor) -> None:
        """
        安全地注册参数
        
        参数:
            name: 参数名称
            tensor: 参数张量
        """
        if not isinstance(tensor,Tensor):
            raise TypeError(f"parameter must be Tensor, but get the {type(tensor)}")
        if not tensor.requires_grad:
            raise ValueError("register parameter must need grad")
        # 获取_parameter
        _parameter = object.__getattribute__(self, '_parameters')
        # 注册参数
        _parameter[name] = tensor
        object.__setattr__(self, name, tensor)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 (batch_size, num_features) 或 (batch_size, num_features, length)
            
        返回:
            归一化后的张量
        """
        if not isinstance(x,Tensor):
            raise TypeError(f"input must be Tensor, but get {type(x)}")
        # 检查输入的形状
        if x.ndim not in [2,3]:
            raise ValueError(f"batchnorm1d expect 2d or 3d, but get {x.ndim} tensor")
        if x.shape[1]!= self.num_features:
            raise ValueError(f"input feature is not match{self.num_features}, but get {x.shape[1]}")
        # 确定归一化的轴
        if x.ndim ==2:
            # 形状：（batch_Size, num_feature)
            axes = (0,)# 沿批次维度进行归一化
        else:
            # 形状：（batch_size, num_feature, length）
            axes = (0,2)

        if self.training:
            return self._forward_train(x,axes)# 使用当前的统计量
        else:
            return self._forward_eval(x,axes)# 使用运行的统计量

    def _forward_train(self, x: Tensor, axes: tuple) -> Tensor:
        """训练模式前向传播"""
        # TODO: 实现前向传播
        # 如果在训练模式，计算当前批次的均值和方差，更新运行统计量
        # 如果在评估模式，使用运行统计量
        # 应用归一化： (x - mean) / sqrt(var + eps)
        # 应用缩放和偏移： gamma * normalized_x + beta

        # self.current_mean = x.mean(axis=axes, keepdims = True)
        # self.current_val = x.var(axis = axes, keepdims = True)

        # 计算当前批次的均值和方差
        mean_result = x.mean(axis=axes, keepdims=True)
        var_result = x.var(axis=axes, keepdims=True)
        
        # 确保我们得到了有效的Tensor对象
        if not isinstance(mean_result, Tensor):
            raise TypeError(f"Expected Tensor, got {type(mean_result)}")
        
        # 使用object.__setattr__直接设置属性，避免__setattr__的干扰
        object.__setattr__(self, 'current_mean', mean_result)
        object.__setattr__(self, 'current_val', var_result)
        
        # 验证赋值是否成功
        current_mean_check = object.__getattribute__(self, 'current_mean')
        if current_mean_check is None:
            raise ValueError("Failed to assign current_mean")
        
        # print(f"DEBUG: Successfully assigned current_mean: {type(current_mean_check)}, {current_mean_check.shape}")

        # 更新计算统计量
        if self.track_running_stats:
            # 使用no_grad上下文管理器
            with Tensor.no_grad():# 运行统计量不参与梯度运算
                self.running_mean.data = (
                    (1-self.momentum)*self.running_mean.data+\
                    self.momentum*self.current_mean.data.reshape(-1)
                )
                self.running_var.data = (
                    (1-self.momentum)*self.running_var.data+\
                    self.momentum*self.current_val.data.reshape(-1)
                )
        # 归一化
        # pdb.set_trace()
        x_normalized = (x - self.current_mean)/(self.current_val + self.eps).sqrt()
        # 应用缩放和偏移
        if self.affine:
            # 重塑权重和偏置
            if x.ndim ==2:
                weight_reshaped = self.weight.reshape(1,-1)
                bias_reshaped = self.bias.reshape(1,-1)
            else:
                weight_reshaped = self.weight.reshape(1,-1,1)
                bias_reshaped = self.bias.reshape(1,-1,1)

            x_normalized = x_normalized * weight_reshaped + bias_reshaped
        return x_normalized
    
    def _forward_eval(self, x: Tensor, axes: tuple) -> Tensor:
        #评估模式的前向传播
        if not self.track_running_stats:
            raise RuntimeError("In eval mode need use track_running_stats")
        if x.ndim ==2:
            running_mean_reshaped = self.running_mean.reshape(1,-1)
            running_var_reshaped = self.running_var.reshape(1,-1)
        else:
            running_mean_reshaped = self.running_mean.reshape(1,-1,1) 
            running_var_reshaped =  self.running_var.reshape(1,-1,1)


        # 归一化
        x_normalized = (x - running_mean_reshaped)/np.sqrt(running_var_reshaped + self.eps)
        # 应用缩放和偏移
        if self.affine:
            # 重塑权重和偏置
            if x.ndim ==2:
                weight_reshaped = self.weight.reshape(1,-1)
                bias_reshaped = self.bias.reshape(1,-1)
            else:
                weight_reshaped = self.weight.reshape(1,-1,1)
                bias_reshaped = self.bias.reshape(1,-1,1)

            x_normalized = x_normalized * weight_reshaped + bias_reshaped
        return x_normalized
    def __call__(self, x: Tensor) -> Tensor:
        """使实例可调用"""
        return self.forward(x)
    
    def train(self):
        """设置为训练模式"""
        # TODO: 设置训练模式
        self.training = True
    
    def eval(self):
        """设置为评估模式"""
        # TODO: 设置评估模式
        self.training = False
    
    def parameters(self):
        """
        返回层的所有参数
        
        返回:
            参数列表
        """
        # TODO: 返回所有可训练参数
        params = []
        if self.affine:
            params.extend([self.weight, self.bias])
        return params

    def reset_parameters(self) -> None:
        """重置参数"""
        if self.affine:
            ones_(self.weight)
            zeros_(self.bias)
        if self.track_running_stats:
            zeros_(self.running_mean)
            ones_(self.running_var)

    def extra_repr(self) -> str:
        """
        返回层的额外描述信息，用于__repr__
        """
        return f"{self.num_features}, eps = {self.eps}, momentum = {self.momentum}, affline = {self.affine},\
        track_running_stats = {self.track_running_stats}"
    
    def __repr__(self) -> str:
        return f"BatchNorm1d({self.extra_repr()})"
    
class BatchNorm2d(Module):
    """
    二维批归一化层
    用于卷积层的批归一化
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, 
                 affine: bool = True, track_running_stats: bool = True):
        """
        初始化二维批归一化层
        
        参数:
            num_features: 特征数（通道数）
            eps: 数值稳定性常数，防止除零
            momentum: 运行均值和方差的动量
            affine: 是否学习缩放和偏移参数
            track_running_stats: 是否跟踪运行统计量
        """
        # TODO: 初始化参数
        # 初始化可学习的缩放和偏移参数
        # 初始化运行均值和方差
        # 设置其他超参数
        super().__init__()
        if num_features <=0:
            raise ValueError(f"num_feature must be postive num, but the num is {num_features}")
        if eps<0:
            raise ValueError(f"eps must be the positive number,but get{eps}")
        if momentum<0 or momentum>1:
            raise ValueError(f"momentum must between [0,1], but get {momentum}")
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # 可学习缩放和偏移参数
        if affine:
            self.weight = Tensor(
                np.ones(num_features, dtype=np.float32),
                requires_grad= True,
                name=f"batchnorm1d_weight_{num_features}"
            )
            self.bias = Tensor(
                np.zeros(num_features,dtype=np.float32),
                requires_grad=True,
                name=f"batchnorm2d_weight_{num_features}"
            )
        else:
            self.weight = None
            self.bias = None

       # 运行统计量（用于评估模式）
        if track_running_stats:
            self.running_mean = Tensor(
                np.zeros(num_features,dtype=np.float32),
                requires_grad=False
            )
            self.running_var = Tensor(
                np.ones(num_features,dtype=np.float32),
                requires_grad=False
            )
        else:
            self.running_mean = None
            self.running_var = None
        # 当前的统计量（训练）
        # self.current_mean = None
        # self.current_val = None
        object.__setattr__(self, 'current_mean', None)
        object.__setattr__(self, 'current_val', None)
        # 评估/训练模式
        self.training = True

        self.reset_parameters()
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 (batch_size, channels, height, width)
            
        返回:
            归一化后的张量
        """
        # TODO: 实现前向传播
        # 处理4D输入，按通道归一化
        # 其他逻辑与BatchNorm1d类似
        if not isinstance(x,Tensor):
            raise TypeError(f"input must be Tensor, but get {type(x)}")
        # 检查输入的形状
        if x.ndim != 4:
            raise ValueError(f"batchnorm1d expect 4d, but get {x.ndim} tensor")
        if x.shape[1]!= self.num_features:
            raise ValueError(f"input feature is not match{self.num_features}, but get {x.shape[1]}")
        # 确定归一化的轴,沿着批次，高度和宽度

        axes = (0,2,3)

        if self.training:
            return self._forward_train(x,axes)# 使用当前的统计量
        else:
            return self._forward_eval(x,axes)# 使用运行的统计量
        
    def _forward_train(self, x: Tensor, axes: tuple) -> Tensor:
        """
        计算当前的批次的均值和方差，训练模式前向传播
        """
        # self.current_mean = x.mean(axis=axes, keepdims = True)
        # self.current_val = x.var(axis = axes, keepdims = True)
        # 计算当前批次的均值和方差
        mean_result = x.mean(axis=axes, keepdims=True)
        var_result = x.var(axis=axes, keepdims=True)
        
        # 确保我们得到了有效的Tensor对象
        if not isinstance(mean_result, Tensor):
            raise TypeError(f"Expected Tensor, got {type(mean_result)}")
        
        # 使用object.__setattr__直接设置属性，避免__setattr__的干扰
        object.__setattr__(self, 'current_mean', mean_result)
        object.__setattr__(self, 'current_val', var_result)
        
        # 验证赋值是否成功
        current_mean_check = object.__getattribute__(self, 'current_mean')
        if current_mean_check is None:
            raise ValueError("Failed to assign current_mean")
        
        # 更新计算统计量
        if self.track_running_stats:
            with Tensor.no_grad():# 运行统计量不参与梯度运算
                self.running_mean.data = (
                    (1-self.momentum)*self.running_mean.data+\
                    self.momentum*self.current_mean.data.reshape(-1)
                )
                self.running_var.data = (
                    (1-self.momentum)*self.running_var.data+\
                    self.momentum*self.current_val.data.reshape(-1)
                )
        # 归一化
        x_normalized = (x - self.current_mean)/(self.current_val + self.eps).sqrt()
        
        # 应用缩放和偏移
        if self.affine:
            # 重塑权重和偏置
            weight_reshaped = self.weight.reshape(1,-1,1,1)
            bias_reshaped = self.bias.reshape(1,-1,1,1)
            x_normalized = x_normalized * weight_reshaped + bias_reshaped

        return x_normalized
    

    def _forward_eval(self, x: Tensor, axes: tuple) -> Tensor:
        """
        评估模式的前向传播
        """
        if not self.track_running_stats:
            raise RuntimeError("In eval mode need use track_running_stats")
   
        running_mean_reshaped = self.running_mean.reshape(1,-1,1,1)
        running_var_reshaped = self.running_var.reshape(1,-1,1,1)

        # 归一化
        x_normalized = (x - running_mean_reshaped)/np.sqrt(running_var_reshaped + self.eps)
        # 应用缩放和偏移
        if self.affine:
            weight_reshaped = self.weight.reshape(1,-1,1,1)
            bias_reshaped = self.bias.reshape(1,-1,1,1)
            x_normalized = x_normalized * weight_reshaped + bias_reshaped
        return x_normalized
    
    def __call__(self, x: Tensor) -> Tensor:
        """使实例可调用"""
        return self.forward(x)
    
    def train(self):
        """设置为训练模式"""
        # TODO: 设置训练模式
        self.track_running_stats = True
    
    def eval(self):
        """设置为评估模式"""
        # TODO: 设置评估模式
        self.track_running_stats = False
    
    def parameters(self):
        """
        返回层的所有参数
        
        返回:
            参数列表
        """
        # TODO: 返回所有可训练参数

        params = []
        if self.affine:
            params.extend([self.weight, self.bias])
        return params

    def reset_parameters(self) -> None:
        """重置参数"""
        if self.affine:
            ones_(self.weight)
            zeros_(self.bias)
        if self.track_running_stats:
            zeros_(self.running_mean)
            ones_(self.running_var)

    def extra_repr(self) -> str:
        """
        返回层的额外描述信息，用于__repr__
        """
        return f"{self.num_features}, eps = {self.eps},\
        momentum = {self.momentum}, affline = {self.affine},\
        track_running_stats = {self.track_running_stats}"
    
    def __repr__(self) -> str:
        return f"BatchNorm2d({self.extra_repr()})"


class ReLU(Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.mask = None  # 保存激活掩码，用于反向传播

    def forward(self, x:Tensor)->Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量
            
        返回:
            输出张量
        """
        return F.relu(x, inplace=self.inplace)
    
    def __call__(self, x: Tensor) -> Tensor:
        """使实例可调用"""
        return self.forward(x)
    
    def parameters(self):
        """
        ReLU 层没有可训练参数
        
        返回:
            空列表
        """
        return []
    
    def extra_repr(self) -> str:
        """
        返回层的额外描述信息，用于 __repr__
        """
        return f'inplace={self.inplace}'
    
    def __repr__(self) -> str:
        return f'ReLU({self.extra_repr()})'

class Sigmoid(Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.mask = None  # 保存激活掩码，用于反向传播

    def forward(self, x:Tensor)->Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量
            
        返回:
            输出张量
        """
        return F.sigmoid(x, inplace=self.inplace)
    
    def __call__(self, x: Tensor) -> Tensor:
        """使实例可调用"""
        return self.forward(x)
    
    def parameters(self):
        """
        ReLU 层没有可训练参数
        
        返回:
            空列表
        """
        return []
    
    def extra_repr(self) -> str:
        """
        返回层的额外描述信息，用于 __repr__
        """
        return f'inplace={self.inplace}'
    
    def __repr__(self) -> str:
        return f'Sigmoid({self.extra_repr()})'
    
class Tanh(Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.mask = None  # 保存激活掩码，用于反向传播

    def forward(self, x:Tensor)->Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量
            
        返回:
            输出张量
        """
        return F.tanh(x, inplace=self.inplace)
    
    def __call__(self, x: Tensor) -> Tensor:
        """使实例可调用"""
        return self.forward(x)
    
    def parameters(self):
        """
        ReLU 层没有可训练参数
        
        返回:
            空列表
        """
        return []
    
    def extra_repr(self) -> str:
        """
        返回层的额外描述信息，用于 __repr__
        """
        return f'inplace={self.inplace}'
    
    def __repr__(self) -> str:
        return f'tanh({self.extra_repr()})'

class LeakyReLU(Module):
    pass

class Conv2d(Module):
    """
    2D卷积层
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size,tuple) else (kernel_size,kernel_size)
        self.stride = stride if isinstance(stride,tuple) else (stride,stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.bias = bias
        # 初始化卷积核权重
        self.weight = Tensor(
            np.random.randn(out_channels,in_channels,self.kernel_size[0],self.kernel_size[1]),
            requires_grad=True,
            name=f"Conv2d_weight_{in_channels}->{out_channels}_{kernel_size}"
        )
        self.register_parameter('weight', self.weight)
        # 初始化偏置
        if bias:
            self.bias_param = Tensor(
                np.zeros(out_channels),
                requires_grad=True,
                name=f"Conv2d_bias_{out_channels}"
            )
            self.register_parameter('bias', self.bias_param)
        else:
            self.bias_param = None
        self.reset_parameters()

    def register_parameter(self, name: str, tensor: Tensor) -> None:
        """
        安全地注册参数
        
        参数:
            name: 参数名称
            tensor: 参数张量
        """
        if not isinstance(tensor,Tensor):
            raise TypeError(f"parameter must be Tensor, but get the {type(tensor)}")
        if not tensor.requires_grad:
            raise ValueError("register parameter must need grad")
        # 获取_parameter
        _parameter = object.__getattribute__(self, '_parameters')
        # 注册参数
        _parameter[name] = tensor
        object.__setattr__(self, name, tensor)
        
    def forward(self, x:Tensor)->Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量
            
        返回:
            输出张量
        """
        return F.conv2d(x, self.weight, self.bias_param, self.stride, self.padding)
    
    def reset_parameters(self):
        kaiming_uniform_(self.weight ,nonlinearity='relu')
        if self.bias_param is not None:
            zeros_(self.bias_param)

    def extra_repr(self) -> str:
        """
        返回层的额外描述信息，用于 __repr__
        """
        return (f"in_channels={self.in_channels},out_channels={self.out_channels},kernel_size={self.kernel_size}\
                stride={self.stride},padding={self.padding},bias={self.bias}")
    
    def __repr__(self) -> str:
        return f'conv2d({self.extra_repr()})'

class MaxPool2d(Module):
    def __init__(self, kernel_size=2, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size,tuple) else (kernel_size,kernel_size)
        self.stride = stride if stride is not None else kernel_size
        self.stride = self.stride if isinstance(self.stride,tuple) else (self.stride,self.stride) 
        self.padding = padding if isinstance(padding,tuple) else (padding, padding)
    def forward(self, x):
        """
        调用functional的函数
        """
        return F.max_pool2d(x,self.kernel_size,self.stride,self.padding)
    def extra_repr(self)->str:
        return f"kernel_size={self.kernel_size},stride={self.stride},padding={self.padding}"
class Flatten(Module):
    """
    展平层
    """
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim 
        self.end_dim = end_dim
        self.input_shape = None
    def forward(self, x:Tensor)->Tensor:
        """
        前向传播
        """
        self.input_shape = x.shape
        # 计算展平之后的形状
        if self.end_dim == -1:
            self.end_dim  = len(x.shape) -1

        # 展平维度
        new_shape =list(x.shape[:self.start_dim])
        flattened_size = 1
        for i in range(self.start_dim, self.end_dim + 1):
            flattened_size *= x.shape[i]
        new_shape.append(flattened_size)
        if self.end_dim<len(x.shape)-1:
            new_shape.extend(x.shape[self.end_dim+1:])

        out = x.reshape(*new_shape)
        def _backward():
            if x.requires_grad:
                grad_out = out.grad
                grad_x = grad_out.reshape(self.input_shape)
                if x.grad is None:
                    x.grad = np.zeros_like(x.data)
                x.grad +=grad_x
        out._backward = _backward
        out._parents = [x]
        return out
    
    def extra_repr(self)->str:
        return f"start_dim={self.start_dim}, end_dim={self.end_dim}"
