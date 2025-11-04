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
            requires_grad = True
        )
      
        self.register_parameter('weight', self.weight)
        # 初始化偏置参数
        if bias:
            self.bias_param = Tensor(
                np.empty(out_features, dtype=np.float32),
                requires_grad=True
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
        super.__init__() #必须调用
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
                requires_grad= True
            )
            self.bias = Tensor(
                np.zeros(num_features,dtype=np.float32),
                requires_grad=True
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
        self.current_mean = None
        self.current_val = None
        # 评估/训练模式
        self.training = True
        self.reset_parameters()
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
        self.current_mean = x.mean(axis=axes, keepdims = True)
        self.current_val = x.var(axis = axes, keepdims = True)

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
        super.__init__() 
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
                requires_grad= True
            )
            self.bias = Tensor(
                np.zeros(num_features,dtype=np.float32),
                requires_grad=True
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
        self.current_mean = None
        self.current_val = None
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
        self.current_mean = x.mean(axis=axes, keepdims = True)
        self.current_val = x.var(axis = axes, keepdims = True)

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

if __name__ == "__main__":
  
    print("Linear 层测试")
    print("=" * 50)
    
    # 测试 1: 基础功能
    print("\n1. 基础功能测试")
    linear = Linear(10, 5, bias=True)
    print(f"创建 Linear 层: {linear}")
    print(f"权重形状: {linear.weight.shape}")
    print(f"偏置形状: {linear.bias_param.shape if linear.bias_param is not None else '无偏置'}")
    
    # 测试 2: 前向传播 (2D 输入)
    print("\n2. 2D 输入前向传播测试")
    x_2d = Tensor(np.random.randn(32, 10), requires_grad=True)  # batch_size=32
    output_2d = linear(x_2d)
    print(f"输入形状: {x_2d.shape}")
    print(f"输出形状: {output_2d.shape}")
    assert output_2d.shape == (32, 5), "2D 输入输出形状错误"
    print("✓ 2D 输入测试通过")
    
    # 测试 3: 前向传播 (1D 输入)
    print("\n3. 1D 输入前向传播测试")
    x_1d = Tensor(np.random.randn(10), requires_grad=True)
    output_1d = linear(x_1d)
    print(f"输入形状: {x_1d.shape}")
    print(f"输出形状: {output_1d.shape}")
    assert output_1d.shape == (5,), "1D 输入输出形状错误"
    print("✓ 1D 输入测试通过")
    
    # 测试 4: 无偏置层
    print("\n4. 无偏置层测试")
    linear_no_bias = Linear(8, 4, bias=False)
    x_test = Tensor(np.random.randn(16, 8))
    output_no_bias = linear_no_bias(x_test)
    print(f"无偏置层输出形状: {output_no_bias.shape}")
    assert output_no_bias.shape == (16, 4), "无偏置层输出形状错误"
    print("✓ 无偏置层测试通过")
    
    # 测试 5: 参数收集
    print("\n5. 参数收集测试")
    params = linear.parameters()
    print(f"参数数量: {len(params)}")
    for i, param in enumerate(params):
        print(f"参数 {i}: 形状={param.shape}, 需要梯度={param.requires_grad}")
    assert len(params) == 2, "参数数量错误"
    print("✓ 参数收集测试通过")
    
    # 测试 6: 错误输入处理
    print("\n6. 错误输入处理测试")
    try:
        linear(Tensor(np.random.randn(32, 8)))  # 错误特征数
        print("✗ 错误特征数未检测到")
    except ValueError as e:
        print(f"✓ 错误特征数检测正常: {e}")
    
    try:
        linear(Tensor(np.random.randn(32, 10, 5)))  # 3D 输入
        print("✗ 3D 输入未检测到")
    except ValueError as e:
        print(f"✓ 3D 输入检测正常: {e}")
    
    print("\n" + "=" * 50)
    print("所有 Linear 层测试通过！🎉")
    """

    """
    print("Dropout 层测试")
    print("=" * 50)
    
    # 测试 1: 基础功能
    print("\n1. 基础功能测试")
    dropout = Dropout(p=0.5)
    print(f"创建 Dropout 层: {dropout}")
    
    # 测试 2: 训练模式
    print("\n2. 训练模式测试")
    x_train = Tensor(np.ones((5, 5)), requires_grad=True)
    print("输入 (全1):")
    print(x_train.data)
    
    output_train = dropout(x_train)
    print("Dropout 输出 (训练模式):")
    print(output_train.data)
    
    # 检查是否应用了 dropout
    unique_values = np.unique(output_train.data)
    print(f"输出中的唯一值: {unique_values}")
    
    # 应该包含 0 和 2 (因为 scale=1/(1-0.5)=2)
    assert 0 in unique_values or 2 in unique_values, "训练模式下未正确应用 dropout"
    print("✓ 训练模式测试通过")
    
    # 测试 3: 评估模式
    print("\n3. 评估模式测试")
    dropout.eval()
    x_eval = Tensor(np.ones((5, 5)), requires_grad=True)
    # pdb.set_trace()
    output_eval = dropout(x_eval)
    print("Dropout 输出 (评估模式):")
    print(output_eval.data)  
    # 在评估模式下，输出应该等于输入
    assert np.allclose(output_eval.data, x_eval.data), "评估模式下输出不等于输入"
    print("✓ 评估模式测试通过")
    
    # 测试 4: p=0 的情况
    print("\n4. p=0 测试")
    dropout_zero = Dropout(p=0)
    dropout_zero.train()
    x_zero = Tensor(np.ones((3, 3)), requires_grad=True)
    output_zero = dropout_zero(x_zero)
    
    # 当 p=0 时，所有元素都应该保留
    assert np.allclose(output_zero.data, x_zero.data), "p=0 时输出不等于输入"
    print("✓ p=0 测试通过")
    
    # 测试 5: p=1 的情况
    print("\n5. p=1 测试")
    dropout_one = Dropout(p=1)
    dropout_one.train()
    x_one = Tensor(np.ones((3, 3)), requires_grad=True)
    output_one = dropout_one(x_one)
    
    # 当 p=1 时，所有元素都应该被置零
    assert np.allclose(output_one.data, 0), "p=1 时输出不全为零"
    print("✓ p=1 测试通过")
    
    # 测试 6: 期望值保持
    print("\n6. 期望值保持测试")
    dropout_test = Dropout(p=0.3)
    dropout_test.train()
    
    # 多次运行，检查期望值
    x_test = Tensor(np.ones(1000), requires_grad=True)
    total = 0
    runs = 100
    
    for _ in range(runs):
        output_test = dropout_test(x_test)
        total += np.mean(output_test.data)
    
    average_mean = total / runs
    print(f"平均输出值: {average_mean:.4f} (期望接近 1.0)")
    
    # 平均值应该接近 1.0 (由于缩放)
    assert 0.95 < average_mean < 1.05, f"期望值不保持，得到 {average_mean}"
    print("✓ 期望值保持测试通过")
    
    # 测试 7: 梯度测试
    print("\n7. 梯度测试")
    dropout_grad = Dropout(p=0.5)
    dropout_grad.train()
    
    x_grad = Tensor(np.random.randn(10, 5), requires_grad=True)
    output_grad = dropout_grad(x_grad)
    
    # 模拟一个损失函数
    loss = output_grad.sum()
    loss.backward()
    
    # 检查梯度是否存在
    assert x_grad.grad is not None, "输入梯度未计算"
    print(f"输入梯度形状: {x_grad.grad.shape}")
    print("✓ 梯度测试通过")
    
    # 测试 8: 模式切换
    print("\n8. 模式切换测试")
    dropout_switch = Dropout(p=0.5)
    
    # 初始应为训练模式
    assert dropout_switch.training == True, "初始模式不是训练模式"
    
    # 切换到评估模式
    dropout_switch.eval()
    assert dropout_switch.training == False, "切换到评估模式失败"
    
    # 切换回训练模式
    dropout_switch.train()
    assert dropout_switch.training == True, "切换回训练模式失败"
    print("✓ 模式切换测试通过")
    
    print("\n" + "=" * 50)
    print("所有 Dropout 层测试通过！🎉")
   
    print("BatchNorm 层测试")
    print("=" * 50)
    
    # 测试 1: BatchNorm1d 基础功能
    print("\n1. BatchNorm1d 基础功能测试")
    bn1d = BatchNorm1d(64)
    print(f"创建 BatchNorm1d: {bn1d}")
    
    # 2D 输入测试
    x_2d = Tensor(np.random.randn(32, 64), requires_grad=True)
    output_2d = bn1d(x_2d)
    print(f"2D 输入形状: {x_2d.shape}")
    print(f"2D 输出形状: {output_2d.shape}")
    assert output_2d.shape == (32, 64), "2D 输入输出形状错误"
    
    # 检查输出统计特性
    output_mean = np.mean(output_2d.data, axis=0)
    output_std = np.std(output_2d.data, axis=0)
    print(f"输出均值范围: [{np.min(output_mean):.3f}, {np.max(output_mean):.3f}]")
    print(f"输出标准差范围: [{np.min(output_std):.3f}, {np.max(output_std):.3f}]")
    
    # 在训练模式下，输出应该接近 N(0,1) 分布
    assert np.allclose(output_mean, 0, atol=0.1), "输出均值不接近 0"
    assert np.allclose(output_std, 1, atol=0.1), "输出标准差不接近 1"
    print("✓ BatchNorm1d 2D 输入测试通过")
    
    # 3D 输入测试
    x_3d = Tensor(np.random.randn(32, 64, 10), requires_grad=True)
    output_3d = bn1d(x_3d)
    print(f"3D 输入形状: {x_3d.shape}")
    print(f"3D 输出形状: {output_3d.shape}")
    assert output_3d.shape == (32, 64, 10), "3D 输入输出形状错误"
    print("✓ BatchNorm1d 3D 输入测试通过")
    
    # 测试 2: BatchNorm1d 训练/评估模式
    print("\n2. BatchNorm1d 训练/评估模式测试")
    
    # 训练模式
    bn1d.train()
    x_train = Tensor(np.ones((16, 64)) * 5, requires_grad=True)  # 常数输入
    output_train = bn1d(x_train)
    print(f"训练模式输出均值: {np.mean(output_train.data):.3f}")
    
    # 评估模式
    bn1d.eval()
    x_eval = Tensor(np.ones((16, 64)) * 5, requires_grad=True)
    output_eval = bn1d(x_eval)
    print(f"评估模式输出均值: {np.mean(output_eval.data):.3f}")
    
    # 训练和评估模式输出应该不同
    assert not np.allclose(output_train.data, output_eval.data), "训练和评估模式输出相同"
    print("✓ BatchNorm1d 模式切换测试通过")
    
    # 测试 3: BatchNorm2d 基础功能
    print("\n3. BatchNorm2d 基础功能测试")
    bn2d = BatchNorm2d(32)
    print(f"创建 BatchNorm2d: {bn2d}")
    
    x_4d = Tensor(np.random.randn(8, 32, 14, 14), requires_grad=True)
    # pdb.set_trace()
    output_4d = bn2d(x_4d)
    print(f"4D 输入形状: {x_4d.shape}")
    print(f"4D 输出形状: {output_4d.shape}")
    assert output_4d.shape == (8, 32, 14, 14), "4D 输入输出形状错误"
    
    # 检查输出统计特性
    output_mean = np.mean(output_4d.data, axis=(0, 2, 3))
    output_std = np.std(output_4d.data, axis=(0, 2, 3))
    print(f"输出均值范围: [{np.min(output_mean):.3f}, {np.max(output_mean):.3f}]")
    print(f"输出标准差范围: [{np.min(output_std):.3f}, {np.max(output_std):.3f}]")
    
    # 在训练模式下，输出应该接近 N(0,1) 分布
    assert np.allclose(output_mean, 0, atol=0.1), "输出均值不接近 0"
    assert np.allclose(output_std, 1, atol=0.1), "输出标准差不接近 1"
    print("✓ BatchNorm2d 测试通过")
    
    # 测试 4: 参数收集
    print("\n4. 参数收集测试")
    bn1d_params = bn1d.parameters()
    bn2d_params = bn2d.parameters()
    
    print(f"BatchNorm1d 参数数量: {len(bn1d_params)}")
    print(f"BatchNorm2d 参数数量: {len(bn2d_params)}")
    
    for i, param in enumerate(bn1d_params):
        print(f"BatchNorm1d 参数 {i}: 形状={param.shape}")
    
    for i, param in enumerate(bn2d_params):
        print(f"BatchNorm2d 参数 {i}: 形状={param.shape}")
    
    assert len(bn1d_params) == 2, "BatchNorm1d 参数数量错误"
    assert len(bn2d_params) == 2, "BatchNorm2d 参数数量错误"
    print("✓ 参数收集测试通过")
    
    # 测试 5: 无仿射变换
    print("\n5. 无仿射变换测试")
    bn_no_affine = BatchNorm1d(32, affine=False)
    x_test = Tensor(np.random.randn(16, 32), requires_grad=True)
    output_no_affine = bn_no_affine(x_test)
    params_no_affine = bn_no_affine.parameters()
    
    print(f"无仿射变换参数数量: {len(params_no_affine)}")
    assert len(params_no_affine) == 0, "无仿射变换时参数数量不为 0"
    print("✓ 无仿射变换测试通过")
    
    # 测试 6: 梯度测试
    print("\n6. 梯度测试")
    bn_grad = BatchNorm1d(16)
    x_grad = Tensor(np.random.randn(8, 16), requires_grad=True)
    output_grad = bn_grad(x_grad)
    
    # 模拟损失函数
    loss = output_grad.sum()
    # pdb.set_trace()
    loss.backward()
    
    # 检查梯度是否存在
    assert x_grad.grad is not None, "输入梯度未计算"
    assert bn_grad.weight.grad is not None, "权重梯度未计算"
    assert bn_grad.bias.grad is not None, "偏置梯度未计算"
    
    print(f"输入梯度形状: {x_grad.grad.shape}")
    print(f"权重梯度形状: {bn_grad.weight.grad.shape}")
    print(f"偏置梯度形状: {bn_grad.bias.grad.shape}")
    print("✓ 梯度测试通过")
    
    print("\n" + "=" * 50)
    print("所有 BatchNorm 层测试通过！🎉")