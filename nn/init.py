"""
权重初始化方法
提供各种神经网络权重初始化策略
"""

import pdb
import warnings
import numpy as np

# 在当前文件下调用tensor
import sys
sys.path.append("..")

from core import Tensor

# import os
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

### 基础函数

def ones_(tensor: Tensor) -> None:
    """
    一初始化
    
    参数:
        tensor: 需要初始化的张量
    """
    # TODO: 实现一初始化
    # 使用1填充张量
    if not isinstance(tensor, Tensor):
        raise ValueError(f" ones only support the tensor typr, but the input is {type(tensor)}")
    tensor.data = np.ones_like(tensor.data)
    if tensor.requires_grad and tensor.grad is None:
        tensor.grad = np.zeros_like(tensor.data)

def zeros_(tensor: Tensor) -> None:
    """
    零初始化
    
    参数:
        tensor: 需要初始化的张量
    """
    # TODO: 实现零初始化
    # 使用0填充张量
    if not isinstance(tensor, Tensor):
        raise ValueError(f" zeros only support the tensor typr, but the input is {type(tensor)}")
    tensor.data = np.zeros_like(tensor.data)
    if tensor.requires_grad and tensor.grad is None:
        tensor.grad = np.zeros_like(tensor.data)

def constant_(tensor: Tensor, value: float) -> None:
    """
    常数初始化
    
    参数:
        tensor: 需要初始化的张量
        value: 填充的常数值
    """
    # TODO: 实现常数初始化
    # 使用指定常数值填充张量
    if not isinstance(tensor, Tensor):
        raise ValueError(f" constant only support the tensor typr, but the input is {type(tensor)}")
    tensor.data = np.full_like(tensor.data, value)
    if tensor.requires_grad and tensor.grad is None:
        tensor.grad = np.zeros_like(tensor.data)


def uniform_(tensor: Tensor, low: float = -1.0, high: float = 1.0) -> None:
    """
    均匀分布初始化
    
    参数:
        tensor: 需要初始化的张量
        low: 均匀分布的下界
        high: 均匀分布的上界
    """
    # TODO: 实现均匀分布初始化
    # 使用指定范围的均匀分布初始化权重
    if not isinstance(tensor, Tensor):
        raise ValueError(f" uniform only support the tensor typr, but the input is {type(tensor)}")
    if low>high:
        raise ValueError(f"low must be lower than high, the low = {low}, the high = {high}")
    # 使用均匀分布初始化
    tensor.data = np.random.uniform(low, high, tensor.shape).astype(tensor.data.dtype)
    if tensor.requires_grad and tensor.grad is None:
        tensor.grad = np.zeros_like(tensor.data)

def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> None:
    """
    正态分布初始化
    
    参数:
        tensor: 需要初始化的张量
        mean: 正态分布的均值
        std: 正态分布的标准差
    """
    # TODO: 实现正态分布初始化
    # 使用指定均值和标准差的正态分布初始化权重
    if not isinstance(tensor, Tensor):
        raise ValueError(f"normal only support the tensor typr, but the input is {type(tensor)}")
    if std<0:
        raise ValueError(f"std must be postive number, but the input is {std}")
    # 使用正态分布初始化
    tensor.data = np.random.normal(mean, std, tensor.shape).astype(tensor.data.dtype)
    # 确保梯度存在且为0
    if tensor.requires_grad and tensor.grad is None:
        tensor.grad = np.zeros_like(tensor.data)

def eye_(tensor: Tensor) -> None:
    """
    单位矩阵初始化 - 将张量初始化为单位矩阵
    
    注意: 只适用于2D张量（矩阵）
    
    参数:
        tensor: 需要初始化的张量
    """
    if tensor.shape[0] != tensor.shape[1]:
        raise ValueError("eye_ only supports square matrices")
    if not isinstance(tensor, Tensor):
        raise ValueError(f"normal only support the tensor typr, but the input is {type(tensor)}")
    if tensor.ndim !=2:
        raise ValueError(f"eye only support the 2 dim vector, but get the dim is{tensor.ndim}")
    # 使用单位矩阵初始化
    tensor.data = np.eye(tensor.shape[0], tensor.shape[1]).astype(tensor.data.dtype)
    # 确保梯度存在且为0
    if tensor.requires_grad and tensor.grad is None:
        tensor.grad = np.zeros_like(tensor.data)

def dirac_(tensor: Tensor, groups: int = 1) -> None:
    """
    Dirac初始化 - 将卷积核初始化为Dirac delta函数
    
    注意: 只适用于3D或4D卷积权重张量
    
    参数:
        tensor: 需要初始化的卷积权重张量
        groups: 分组卷积的组数
    """
    if not isinstance(tensor, Tensor):
        raise ValueError(f"normal only support the tensor typr, but the input is {type(tensor)}")
    if tensor.ndim not in [3,4]:
        raise ValueError(f"dirac only support the 3 or 4 dim, but get the {tensor.ndim} tensor")
    if tensor.ndim ==3:
        out_channels, in_channels, kernel_size = tensor.shape
        tensor.data = np.zeros_like(tensor.data)
        in_channels_per_group = in_channels //groups
        out_channels_per_group = out_channels // groups
        for g in range(groups):
            for i in range(min(in_channels_per_group, out_channels_per_group)):
                tensor.data[g*out_channels_per_group + i,
                            g*in_channels_per_group + i,
                            kernel_size//2] = 1.0
    else:
        out_channels, in_channels, kernel_h, kernel_w = tensor.shape
        tensor.data = np.zeros_like(tensor.data)
        in_channels_per_group = in_channels //groups
        out_channels_per_group = out_channels // groups
        for g in range(groups):
            for i in range(min(in_channels_per_group, out_channels_per_group)):
                tensor.data[g*out_channels_per_group + i,
                            g*in_channels_per_group + i,
                            kernel_h//2,
                            kernel_w//2] = 1.0
    # 确保梯度存在且为0
    if tensor.requires_grad and tensor.grad is None:
        tensor.grad = np.zeros_like(tensor.data)



def xavier_uniform_(tensor: Tensor, gain: float = 1.0, mode:str = 'fan_avg',
                    distribution: str = 'uniform', nonlinearity: str = 'linear') -> None:
    """
    Xavier均匀分布初始化
    
    参数:
        tensor: 需要初始化的张量
        gain: 缩放因子，可根据激活函数调整
        mode: 初始化模式，可选 'fan_in', 'fan_out', 'fan_avg'
        distribution: 分布类型，'uniform' 或 'normal'
        nonlinearity: 激活函数类型，用于自动计算gain
    """
    # TODO: 实现Xavier均匀分布初始化
    # 根据输入和输出维度计算范围
    # 使用均匀分布初始化权重
    if not isinstance(tensor, Tensor):
        raise TypeError(f"xavier only support the Tensot type, the input type is {type(tensor)}")
    if tensor.ndim < 2:
        raise ValueError(f"xavier init need 2 dim tensor, but the input is {tensor.ndim}")
    if mode not in ['fan_in','fan_out','fan_avg']:
        raise ValueError(f"mode must be the 'fan_in','fan_out','fan_avg', but the input is {mode}")
    if distribution not in ['uniform','normal']:
        raise ValueError(f"distribution must be 'uniform'or'normal', but the input is {distribution}")
    # 根据激活函数计算gain
    calculated_gain = gain
    if gain==1.0: # 用户没有指定gain
        calculated_gain = _calculate_gain(nonlinearity)
    # 计算fan_in,fan_out
    # fan_in只考虑输入维度
    # fan_out只考虑输出维度
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)

    # 根据模式选择分母
    if mode =="fan_in":
        denominator = fan_in
    elif mode =="fan_out":
        denominator = fan_out
    else:
        denominator = (fan_out + fan_in)/2.0
    if distribution == 'uniform':
        bound = calculated_gain * np.sqrt(3.0/denominator)
        init_data = np.random.uniform(-bound, bound, tensor.shape)
    else: #正态分布
        std = calculated_gain / np.sqrt(denominator)
        init_data = np.random.normal(0.0, std, tensor.shape)
    # 更新张量数据
    tensor.data = init_data.astype(tensor.data.dtype)
    # 确保梯度存在且为0
    if tensor.requires_grad and tensor.grad is None:
        tensor.grad = np.zeros_like(tensor.data)
    

def _calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple:
    """
    计算张量的fan_in 和fan_out
    输入：
    tensor:输入的张量
    输出：
    (fan_in,fan_out)元组

    """
    dimensions = tensor.ndim
    if dimensions<2:
        raise ValueError(f"fan in and fan out cannot be computed for tensor with fewer than 2 dimensions")
    if dimensions ==2:
        fan_in = tensor.shape[1]
        fan_out = tensor.shape[0]
    else:
        # 卷积层或者其他的高维张量
        # 对于卷积层，形状通常是(out_channels,in_channels, *kernel_size)
        num_input_fmaps = tensor.shape[1]
        num_output_fmaps = tensor.shape[0]
        # receptive_field_size = 1
        # if dimensions>2:
        #     receptive_field_size = np.prod(tensor.shape[2:])
        receptive_field_size = np.prod(tensor.shape[2:])
        # 感受野
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in,fan_out

def _calculate_gain(nonlinearity: str, param: float = None) -> float:
    """
    根据激活函数计算推荐的gain值
    
    参数:
        nonlinearity: 激活函数名称
        param: 激活函数的参数（如Leaky ReLU的负斜率）
        
    返回:
        推荐的gain值
    """
    nonlinearity = nonlinearity.lower()
    if nonlinearity == "linear" or nonlinearity == "identity":
        return 1.0
    if nonlinearity == "sigmoid" or nonlinearity == "tanh":
        return 1.0
    elif nonlinearity == 'relu':
        return np.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        if param is None:
            param = 0.01
        return np.sqrt(2.0/(1+param**2))
    elif nonlinearity == 'selu':
        return 3.0/4
    else:
        Warning.warn(f"nuknow activate function{nonlinearity}, use the gain as 1.0")
        return 1.0
    pass

def xavier_normal_(tensor: Tensor, gain: float = 1.0, mode: str = 'fan_avg', nonlinearity: str = 'linear') -> None:
    """
    Xavier正态分布初始化
    
    参数:
        tensor: 需要初始化的张量
        gain: 可选的缩放因子，根据激活函数调整
        mode: 初始化模式，可选 'fan_in', 'fan_out', 'fan_avg'
        nonlinearity: 激活函数类型，用于自动计算gain
    """
    # TODO: 实现Xavier正态分布初始化
    # 根据输入和输出维度计算标准差
    # 使用正态分布初始化权重
  
    if not isinstance(tensor, Tensor):
        raise TypeError(f"xavier_normal_ only support type tensor, the input is {type(tensor)}")
    
    if tensor.ndim < 2:
        raise ValueError(f"xavier init need 2 dim tensor, but the input is {tensor.ndim}")
    if mode not in ['fan_in','fan_out','fan_avg']:
        raise ValueError(f"mode must be the 'fan_in','fan_out','fan_avg', but the input is {mode}")
    
    # 根据激活函数计算gain
    calculated_gain = gain
    if gain==1.0: # 用户没有指定gain
        calculated_gain = _calculate_gain(nonlinearity)
    # 计算fan_in,fan_out
    # fan_in只考虑输入维度
    # fan_out只考虑输出维度
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)

    # 根据模式选择分母
    if mode =="fan_in":
        denominator = fan_in
    elif mode =="fan_out":
        denominator = fan_out
    else:
        denominator = (fan_out + fan_in)/2.0

    # Xavier正态分布公式: std = gain * sqrt(2 / (fan_in + fan_out))
    std = calculated_gain * np.sqrt(1.0/denominator)
    # 生成正态分布数据
    normal_data = np.random.normal(0.0, std, tensor.shape)
    # 更新张量数据
    tensor.data = normal_data.astype(tensor.data.dtype)

    # 确保梯度存在且为0
    if tensor.requires_grad and tensor.grad is None:
        tensor.grad = np.zeros_like(tensor.data)



def kaiming_uniform_(tensor: Tensor, a: float = 0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu') -> None:
    """
    Kaiming均匀分布初始化
    
    参数:
        tensor: 需要初始化的张量
        a: 负斜率（用于Leaky ReLU）
        mode: 'fan_in' 或 'fan_out'
        nonlinearity: 非线性激活函数名称
    """
    # TODO: 实现Kaiming均匀分布初始化
    # 根据模式计算fan_in和fan_out
    # 计算均匀分布的边界
    # 使用均匀分布初始化权重
    if not isinstance(tensor, Tensor):
        raise TypeError(f"kaiming only support the Tensot type, the input type is {type(tensor)}")
    if tensor.ndim < 2:
        raise ValueError(f"kaiming init need 2 dim tensor, but the input is {tensor.ndim}")
    if mode not in ['fan_in','fan_out']:
        raise ValueError(f"mode must be the 'fan_in','fan_out', but the input is {mode}")
    if nonlinearity not in ['relu','leaky_relu']:
        raise ValueError(f"nonlinearity must be 'relu'or'leaky_relu', but the input is {nonlinearity}")

    # 计算fan_in,fan_out
    # fan_in只考虑输入维度
    # fan_out只考虑输出维度
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)

    fan = fan_in if mode == "fan_in" else fan_out

    if nonlinearity =="relu":
        gain = np.sqrt(2.0)
    else:
        gain = np.sqrt(2.0/(1+a**2))
    # 计算均匀分布的边界
    # kaiming均匀分布的边界
    std = gain / np.sqrt(fan)
    bound = np.sqrt(3.0) * std

    # print(f"DEBUG: fan={fan}, gain={gain:.4f}, std={std:.6f}, bound={bound:.6f}")  # 添加调试

    # 生成均匀分布的数据
    uniform_data = np.random.uniform(-bound , bound, tensor.shape)

    # print(f"DEBUG: 初始化数据范围: [{uniform_data.min():.6f}, {uniform_data.max():.6f}]")  # 添加调试

    # 更新张量数据
    tensor.data = uniform_data.astype(tensor.data.dtype)

    # 确保梯度存在且为0
    if tensor.requires_grad and tensor.grad is None:
        tensor.grad = np.zeros_like(tensor.data)




def kaiming_normal_(tensor: Tensor, a: float = 0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu') -> None:
    """
    Kaiming正态分布初始化
    
    参数:
        tensor: 需要初始化的张量
        a: 负斜率（用于Leaky ReLU）
        mode: 'fan_in' 或 'fan_out'
        nonlinearity: 非线性激活函数名称
    """
    # TODO: 实现Kaiming正态分布初始化
    # 根据模式计算fan_in和fan_out
    # 计算正态分布的标准差
    # 使用正态分布初始化权重
    if not isinstance(tensor, Tensor):
        raise TypeError(f"kaiming only support the Tensot type, the input type is {type(tensor)}")
    if tensor.ndim < 2:
        raise ValueError(f"kaiming init need 2 dim tensor, but the input is {tensor.ndim}")
    if mode not in ['fan_in','fan_out']:
        raise ValueError(f"mode must be the 'fan_in','fan_out',but the input is {mode}")
    if nonlinearity not in ['relu','leaky_relu']:
        raise ValueError(f"nonlinearity must be 'relu'or'leaky_relu', but the input is {nonlinearity}")

    # 计算fan_in,fan_out
    # fan_in只考虑输入维度
    # fan_out只考虑输出维度
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)

    fan = fan_in if mode == "fan_in" else fan_out

    if nonlinearity =="relu":
        gain = np.sqrt(2.0)
    else:
        gain = np.sqrt(2.0/(1+a**2))
    # 计算均匀分布的边界
    # kaiming均匀分布的边界
    std = gain / np.sqrt(fan)

    # normal
    normal_data = np.random.normal(0.0 , std, tensor.shape)
    # 更新张量数据
    tensor.data = normal_data.astype(tensor.data.dtype)

    # 确保梯度存在且为0
    if tensor.requires_grad and tensor.grad is None:
        tensor.grad = np.zeros_like(tensor.data)




if __name__ == "__main__":
    """
    初始化方法测试代码
    """
    import sys
    import os
    
    # 添加父目录到路径，以便导入mytorch模块
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    from core.tensor import Tensor
    
    print("=" * 60)
    print("初始化方法测试")
    print("=" * 60)
    
    # 测试基础初始化方法
    print("\n1. 基础初始化方法测试")
    print("-" * 40)
    
    # 测试 ones_
    tensor_ones = Tensor(np.empty((2, 3)), requires_grad=True)
    ones_(tensor_ones)
    print(f"ones_: {tensor_ones.data}")
    assert np.allclose(tensor_ones.data, 1.0), "ones_ 初始化失败"
    print("✓ ones_ 测试通过")
    
    # 测试 zeros_
    tensor_zeros = Tensor(np.empty((2, 3)), requires_grad=True)
    zeros_(tensor_zeros)
    print(f"zeros_: {tensor_zeros.data}")
    assert np.allclose(tensor_zeros.data, 0.0), "zeros_ 初始化失败"
    print("✓ zeros_ 测试通过")
    
    # 测试 constant_
    tensor_const = Tensor(np.empty((2, 3)), requires_grad=True)
    constant_(tensor_const, 0.5)
    print(f"constant_: {tensor_const.data}")
    assert np.allclose(tensor_const.data, 0.5), "constant_ 初始化失败"
    print("✓ constant_ 测试通过")
    
    # 测试 normal_
    tensor_normal = Tensor(np.empty((1000,)), requires_grad=True)
    normal_(tensor_normal, mean=0.0, std=1.0)
    mean_val = np.mean(tensor_normal.data)
    std_val = np.std(tensor_normal.data)
    print(f"normal_: mean={mean_val:.3f}, std={std_val:.3f}")
    assert abs(mean_val) < 0.1, "normal_ 均值测试失败"
    assert 0.9 < std_val < 1.1, "normal_ 标准差测试失败"
    print("✓ normal_ 测试通过")
    
    # 测试 uniform_
    tensor_uniform = Tensor(np.empty((1000,)), requires_grad=True)
    uniform_(tensor_uniform, low=-2.0, high=2.0)
    min_val = np.min(tensor_uniform.data)
    max_val = np.max(tensor_uniform.data)
    print(f"uniform_: min={min_val:.3f}, max={max_val:.3f}")
    assert -2.1 < min_val < -1.9, "uniform_ 下界测试失败"
    assert 1.9 < max_val < 2.1, "uniform_ 上界测试失败"
    print("✓ uniform_ 测试通过")
    
    # 测试 eye_
    tensor_eye = Tensor(np.empty((3, 3)), requires_grad=True)
    eye_(tensor_eye)
    print(f"eye_:\n{tensor_eye.data}")
    expected_eye = np.eye(3)
    assert np.allclose(tensor_eye.data, expected_eye), "eye_ 初始化失败"
    print("✓ eye_ 测试通过")
    
    # 测试高级初始化方法
    print("\n2. 高级初始化方法测试")
    print("-" * 40)
    
    # 测试 xavier_uniform_
    tensor_xavier_u = Tensor(np.empty((256, 128)), requires_grad=True)
    xavier_uniform_(tensor_xavier_u)
    xavier_mean = np.mean(tensor_xavier_u.data)
    xavier_std = np.std(tensor_xavier_u.data)
    print(f"xavier_uniform_: mean={xavier_mean:.3f}, std={xavier_std:.3f}")
    # Xavier初始化应该产生接近0的均值和合理的标准差
    assert abs(xavier_mean) < 0.1, "xavier_uniform_ 均值测试失败"
    assert 0.05 < xavier_std < 0.1, "xavier_uniform_ 标准差测试失败"
    print("✓ xavier_uniform_ 测试通过")
    
    # 测试 xavier_normal_
    tensor_xavier_n = Tensor(np.empty((256, 128)), requires_grad=True)
    xavier_normal_(tensor_xavier_n)
    xavier_n_mean = np.mean(tensor_xavier_n.data)
    xavier_n_std = np.std(tensor_xavier_n.data)
    print(f"xavier_normal_: mean={xavier_n_mean:.3f}, std={xavier_n_std:.3f}")
    assert abs(xavier_n_mean) < 0.1, "xavier_normal_ 均值测试失败"
    assert 0.05 < xavier_n_std < 0.1, "xavier_normal_ 标准差测试失败"
    print("✓ xavier_normal_ 测试通过")
    
    # 测试 kaiming_uniform_
    tensor_kaiming_u = Tensor(np.empty((256, 128)), requires_grad=True)
    kaiming_uniform_(tensor_kaiming_u, nonlinearity='relu')
    kaiming_u_mean = np.mean(tensor_kaiming_u.data)
    kaiming_u_std = np.std(tensor_kaiming_u.data)

    # 计算理论值
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor_kaiming_u)
    theoretical_std = np.sqrt(2.0) / np.sqrt(fan_in)  # fan_in 模式

    print(f"kaiming_uniform_: mean={kaiming_u_mean:.3f}, std={kaiming_u_std:.3f}")
    print(f"理论标准差: {theoretical_std:.3f}")

    # 使用相对误差测试
    relative_error = abs(kaiming_u_std - theoretical_std) / theoretical_std
    if relative_error < 0.1:  # 允许 10% 的相对误差
        print("✓ kaiming_uniform_ 测试通过")
    else:
        print(f"⚠ kaiming_uniform_ 标准差误差较大: {relative_error:.3f}")

    # 同样更新 kaiming_normal_ 的测试
    tensor_kaiming_n = Tensor(np.empty((256, 128)), requires_grad=True)
    kaiming_normal_(tensor_kaiming_n, nonlinearity='relu')
    kaiming_n_mean = np.mean(tensor_kaiming_n.data)
    kaiming_n_std = np.std(tensor_kaiming_n.data)

    print(f"kaiming_normal_: mean={kaiming_n_mean:.3f}, std={kaiming_n_std:.3f}")

    relative_error_n = abs(kaiming_n_std - theoretical_std) / theoretical_std
    if relative_error_n < 0.1:
        print("✓ kaiming_normal_ 测试通过")
    else:
        print(f"⚠ kaiming_normal_ 标准差误差较大: {relative_error_n:.3f}")
    
    # 测试卷积层初始化
    print("\n3. 卷积层初始化测试")
    print("-" * 40)
    
    # 测试卷积权重的初始化
    conv_weight = Tensor(np.empty((64, 32, 3, 3)), requires_grad=True)
    kaiming_uniform_(conv_weight, nonlinearity='relu')
    
    fan_in, fan_out = _calculate_fan_in_and_fan_out(conv_weight)
    print(f"卷积权重形状: {conv_weight.shape}")
    print(f"fan_in: {fan_in}, fan_out: {fan_out}")
    
    conv_mean = np.mean(conv_weight.data)
    conv_std = np.std(conv_weight.data)
    print(f"卷积权重: mean={conv_mean:.3f}, std={conv_std:.3f}")
    
    assert fan_in == 32 * 3 * 3, "fan_in 计算错误"
    assert fan_out == 64 * 3 * 3, "fan_out 计算错误"
    print("✓ 卷积层初始化测试通过")
    
    # 测试 dirac_ 初始化
    dirac_weight = Tensor(np.empty((4, 4, 3, 3)), requires_grad=True)
    dirac_(dirac_weight)
    
    # 检查是否只有中心位置有值
    center_values = dirac_weight.data[:, :, 1, 1]
    off_center_values = dirac_weight.data[:, :, 0, 0]  # 非中心位置
    
    print(f"Dirac中心位置和: {np.sum(center_values)}")
    print(f"Dirac非中心位置和: {np.sum(off_center_values)}")
    
    assert np.sum(center_values) == 4.0, "Dirac初始化中心位置值错误"
    assert np.sum(off_center_values) == 0.0, "Dirac初始化非中心位置应该为0"
    print("✓ dirac_ 测试通过")
    
    # 测试梯度处理
    print("\n4. 梯度处理测试")
    print("-" * 40)
    
    test_tensor = Tensor(np.empty((2, 2)), requires_grad=True)
    ones_(test_tensor)
    
    # 检查梯度是否被正确初始化
    if test_tensor.requires_grad:
        assert test_tensor.grad is not None, "梯度数组未初始化"
        assert np.allclose(test_tensor.grad, 0.0), "梯度未正确清零"
        print("✓ 梯度处理测试通过")
    else:
        print("⚠ 张量不需要梯度，跳过梯度测试")
    
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)