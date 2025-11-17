import numpy as np
import sys
sys.path.append("..")

from core.tensor import Tensor
from typing import  Dict,List,Callable
from nn import model


def get_model_parameters(model):
    """安全地获取模型的所有参数"""
    parameters = []
    
    def _get_layers(module):
        """递归获取所有层"""
        layers = []
        
        # 检查不同的层存储方式
        if hasattr(module, 'layers'):
            layers.extend(module.layers)
        elif hasattr(module, '_layers'):
            layers.extend(module._layers)
        elif hasattr(module, '_modules'):
            layers.extend(module._modules.values())
        else:
            # 如果模块本身是一个层，添加它
            if hasattr(module, 'weight') or hasattr(module, 'bias'):
                layers.append(module)
        
        return layers
    
    def _collect_parameters(module):
        """收集模块的参数"""
        if hasattr(module, 'weight') and module.weight is not None:
            parameters.append(('weight', module.weight))
        if hasattr(module, 'bias') and module.bias is not None:
            parameters.append(('bias', module.bias))
    
    # 处理顶层模型
    layers = _get_layers(model)
    
    # 收集所有层的参数
    for layer in layers:
        _collect_parameters(layer)
        # 递归处理子层
        sub_layers = _get_layers(layer)
        for sub_layer in sub_layers:
            _collect_parameters(sub_layer)
    
    return parameters


class L2Regularizer:
    # l2 正则化
    def __init__(self, coefficient=1e-4):
        self.coefficient = coefficient
    def __call__(self, model) -> float:
        # 计算l2正则化损失
        l2_loss = 0.0
        # for layer in model.layers:
            # if hasattr(layer, 'weight') and layer.weight is not None:
            #     l2_loss += np.sum(layer.weight.data ** 2)
            # if hasattr(layer, 'bias') and layer.bias is not None:
            #     l2_loss += np.sum(layer.bias.data ** 2)
        parameters = get_model_parameters(model)
        for param_name, param in parameters:
            l2_loss += np.sum(param.data ** 2)
        return self.coefficient * l2_loss
    
class FLOPsCalculator:
    # FLOPs计算器
    @staticmethod
    def count_linear(model, input_shape, output_shape)->int:
        """
        计算全连接的FLOPs
        """
        batch_size = input_shape[0]
        input_features = input_shape[1]
        output_features = output_shape[1]
        # 前向：2*input_features*output_features*batch_size
        # 反向：3*input_features*output_features*batch_size
        flops_forward = 2*input_features*output_features*batch_size
        return flops_forward
    
    @staticmethod
    def count_conv2d(module, input_shape, output_shape) ->int:
        """
        计算卷积的FLOPs
        """
        batch_size, in_c, in_h, in_w = input_shape
        out_c, out_h, out_w = output_shape[1],output_shape[2],output_shape[3]
       
        if hasattr(module, 'kernel_size'):
            k_h, k_w = module.kernel_size
        else:
            k_h, k_w = 3, 3  # 默认值

        # 每个输出的位置的FLOPs:in_c*k_h*k_w*2
        flops_per_position = in_c * k_h * k_w * 2
        total_flops = flops_per_position * out_c * out_h * out_w * batch_size
        return total_flops
    
    @staticmethod
    def count_relu(input_shape)->int:
        """
        计算Relu的FLOPs
        """
        return np.prod(input_shape)
    
    @staticmethod
    def count_model_flops(model, input_shape)->Dict[str,int]:
        """
        计算整个模型的FLOPs
        """
        flops_dict = {}
        # 需要使用hook机制获得每层的输入输出形状
        # 这里做了简化，假设我们知道模型结构
        if hasattr(model, 'get_flops_breakdown'): #模型貌似没有这个参数
            return model.get_flops_breakdown(input_shape)
        return flops_dict

class FLOPsRegularizer:
    # 基于FLOPs的正则化
    def __init__(self, coefficient=1e-9,flops_calculator=None):
        self.coefficient = coefficient
        self.flops_calculator = flops_calculator or FLOPsCalculator()

    def __call__(self, model, input_shape):
        """
        基于FLOPs损失模块
        """
        total_flops = 0
        flops_dict = self.flops_calculator.count_model_flops(model, input_shape)

        for layer_name, flops in flops_dict.items():
            total_flops += flops
        return self.coefficient*total_flops
    
class EnergyAwareRegularizer:
    # 能量感知正则化
    def __init__(self, energy_coeff=1e-8,sparsity_coeff=1e-3):
        self.energy_coeff = energy_coeff
        self.sparsity_coeff = sparsity_coeff
    
    def __call__(self, model, activations_dict=None)->float:
        """
        结合能量和稀疏度正则化
        """

        energy_loss = 0.0
        sparsity_loss = 0.0

        # 使用安全的参数获取方式
        parameters = get_model_parameters(model)

        # for layer in model.layers:
        #     if hasattr(layer, 'weight') and layer.weight is not None:
        #         # 简单的能量模型：参数数量*操作类型的系数
        #         param_energy = np.prod(layer.weight.shape) * self._get_energy_factor(layer.weight) #这个函数没有定义
        #         energy_loss += param_energy

        for param_name, param in parameters:
            if 'weight' in param_name: # 只计算权重的能量
                param_energy = np.prod(param.shape) * self._get_energy_factor(param)
                energy_loss += param_energy

        # 稀疏度正则化
        if activations_dict:
            for layer_name, activation in activations_dict.items():
                sparsity = self._calculate_sparsity(activation) #这个函数没有定义
                sparsity_loss += (1-sparsity) #鼓励稀疏性

        return self.energy_coeff * energy_loss + self.sparsity_coeff*sparsity_loss
    
    def _get_energy_factor(self, param)->float:
        # 根据参数类型返回能量参数
        if len(param.shape)==4:# 卷积核
            return 2.0#卷积操作耗能
        elif len(param.shape)==2:#全连接
            return 1.0
        else:
            return 0.1
        
        
    def _calculate_sparsity(self, activation)->float:
        # 计算激活稀疏度
        if isinstance(activation, Tensor):
            data = activation.data
        else:
            data = activation
        # 定义接近0的阈值

        zero_count = np.sum(np.abs(data)<1e-8)
        total_elements = np.prod(data.shape)
        return zero_count/total_elements if total_elements>0 else 0.0

class CombinedRegularizer:
    # 组合正则化
    def __init__(self,regularizers:List[Callable]):
        self.regularizers = regularizers
    def __call__(self, *args, **kwargs)->float:
        total_loss = 0.0
        for regularizers in self.regularizers:
            total_loss += regularizers(*args, **kwargs)
        return total_loss
    

