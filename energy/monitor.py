import numpy as np
import time
from typing import Dict, List, Any
import sys
sys.path.append("..")
from core.tensor import Tensor
from collections import defaultdict


class EnergyMonitor:
    # 能量监控
    def __init__(self, model):
        self.model = model
        self.energy_log = []
        self.hooks = []
        self.activation_stats = {}
        self.tensor_names = {} #存储张量名称映射
        self.flops_counter = FLOPsCounter()
        
    # def _forward_hook(self, module, input, output):
    #     """
    #     前向传播hook,收集能量指针
    #     """
    #     module_name = module.__class__.__name__
    #     module_id = id(module)
    #     # 记录输入张量名称
    #     input_shapes = [x.shape for x in input] if isinstance(input, tuple) else [input.shape]
        
    #     # 记录输出张量的名称
    #     output_name = output.name if hasattr(output, 'name') else "unknown"
    #     output_shape = output.shape

    #     # 估计该层的计算能量
    #     energy_estimate = self._estimate_layer_energy(module, input, output)

    #     # 计算稀疏度
    #     sparsity = self._calculate_sparsity(output)

    #     # 计算FLOPs
    #     flops = self.flops_counter.estimate_layer_flops(module, input_shapes[0], output_shape)
    #     # 记录统计信息
    #     stats = {
    #         'energy': energy_estimate,
    #         'sparsity': sparsity,
    #         'flops': flops,
    #         'input_shapes': input_shapes,
    #         'output_shape': output_shape,
    #         'module_name': module_name,
    #         'module_id': module_id,
    #         'timestamp': time.time(),
    #         'memory_usage': self._estimate_memory_usage(output)  # 估计内存使用
    #     }
    #     self.energy_log.append((module_name,stats))
    #     self.activation_stats[f"{module_name}_{module_id}"] = stats

    #     # 存储张量名称映射
    #     if hasattr(output, 'name'):
    #         self.tensor_name[output.name] = {
    #             'module': module_name,
    #             'module_id':module_id,
    #             'shape': output.shape,
    #             'timestamp': time.time()
    #         }

    def _forward_hook(self, module, input):
        # 前向传播hook，能量收集和稀疏度统计
        self._current_input = input
        self._current_module = module
        return input
    

    def _post_forward_hook(self, module, input,output):
        """
        后向处理hook，在forward完成之后执行
        """
        module_name = module.__class__.__name__
        module_id = id(module)
        
        # 记录输入输出信息
        input_data = input[0] if isinstance(input, tuple) and len(input) > 0 else input
        input_shapes = [x.shape for x in input] if isinstance(input, tuple) else [input.shape]
        output_shape = output.shape
        
        # 估计该层的计算能量
        energy_estimate = self._estimate_layer_energy(module, input_data, output)
        
        # 计算稀疏度
        sparsity = self._calculate_sparsity(output)
        
        # 计算FLOPs
        flops = self.flops_counter.estimate_layer_flops(module, input_shapes[0], output_shape)
        
        # 记录统计信息
        stats = {
            'energy': energy_estimate,
            'sparsity': sparsity,
            'flops': flops,
            'input_shapes': input_shapes,
            'output_shape': output_shape,
            'module_name': module_name,
            'module_id': module_id,
            'timestamp': time.time(),
            'memory_usage': self._estimate_memory_usage(output)
        }
        
        self.energy_log.append((module_name, stats))
        self.activation_stats[f"{module_name}_{module_id}"] = stats
        
        # 存储张量名称映射
        if hasattr(output, 'name'):
            self.tensor_names[output.name] = {
                'module': module_name,
                'module_id': module_id,
                'shape': output.shape,
                'timestamp': time.time()
            }
        
        return output


    def _estimate_layer_energy(self, module, input, output)->float:
        """估计单层消耗的能量 - 确保返回正值"""
        energy = 10.0  # 基础能量值，确保最小能量 > 0
        
        # 如果有权重，增加权重能量
        if hasattr(module, 'weight') and module.weight is not None:
            weight_shape = module.weight.shape if isinstance(module.weight, Tensor) else module.weight.shape
            energy += np.prod(weight_shape) * 1.0
        
        # 根据层类型增加计算能量
        if hasattr(module, 'in_features'):  # Linear层
            energy += module.in_features * module.out_features * 0.2
        elif hasattr(module, 'in_channels'):  # Conv层
            energy += module.in_channels * module.out_channels * 0.5
        
        return energy
    
    def _calculate_sparsity(self, activation) -> float:
        """计算激活稀疏度"""
        if isinstance(activation, Tensor):
            data = activation.data
        else:
            data = activation
            
        zero_threshold = 1e-6
        zero_count = np.sum(np.abs(data) < zero_threshold)
        total_elements = np.prod(data.shape)
        
        return zero_count / total_elements if total_elements > 0 else 0.0
    
    def _estimate_memory_usage(self, tensor) -> int:
        """估计张量内存使用（字节）"""
        if isinstance(tensor, Tensor):
            shape = tensor.shape
        else:
            shape = tensor.shape
            
        # 假设float32（4字节）
        return np.prod(shape) * 4
    
    def attach(self):
        """附加监控hook到模型"""
        # 为模型本身添加hook
        hook = self.model.register_forward_hook(self._post_forward_hook)
        # self.hooks.append(hook)
        self.hooks.append((self.model, hook))
        
        # 为每个子模块添加hook
        for name, module in self.model.named_modules():
            if name != '':  # 跳过根模块，因为已经添加了
                hook = module.register_forward_hook(self._post_forward_hook)
                self.hooks.append((self.model, hook))
                

    def detach(self):
        """移除所有hook"""
        # 安全地移除所有hook，忽略不存在的hook
        for module, hook in self.hooks:
            try:
                _forward_hooks = object.__getattribute__(module, '_forward_hooks')
                if hook in _forward_hooks:
                    _forward_hooks.remove(hook)
                # module.remove_forward_hook(hook)  # 改为上面的手动移除方式，更可控
            except (ValueError, AttributeError):
                # Hook不存在或模块没有_forward_hooks，忽略
                pass
        
        self.hooks.clear()

    def get_total_energy(self) -> float:
        """获取总能量估计"""
        return sum(stats['energy'] for _, stats in self.energy_log)
    
    def get_total_flops(self) -> float:
        """获取总FLOPs"""
        return sum(stats['flops'] for _, stats in self.energy_log)
    
    def get_average_sparsity(self) -> float:
        """获取平均稀疏度"""
        if not self.activation_stats:
            return 0.0
        return np.mean([stats['sparsity'] for stats in self.activation_stats.values()])
    

    def get_memory_usage(self) -> int:
        """获取总内存使用估计"""
        return sum(stats['memory_usage'] for _, stats in self.energy_log)
    
    def get_energy_breakdown(self) -> Dict[str, float]:
        """获取能量分解"""
        breakdown = defaultdict(float)
        for module_name, stats in self.energy_log:
            breakdown[module_name] += stats['energy']
        return dict(breakdown)
    
    def generate_report(self) -> Dict[str, Any]:
        """生成能量监控报告"""
        return {
            'total_energy': self.get_total_energy(),
            'total_flops': self.get_total_flops(),
            'average_sparsity': self.get_average_sparsity(),
            'memory_usage_bytes': self.get_memory_usage(),
            'energy_breakdown': self.get_energy_breakdown(),
            'layer_count': len(self.activation_stats),
            'timestamp': time.time()
        }
    
    def reset(self):
        """重置监控器"""
        self.energy_log.clear()
        self.activation_stats.clear()
        self.tensor_names.clear()


class SparsityMonitor:
    """稀疏度监控器"""
    
    def __init__(self):
        self.sparsity_history = []
        self.activation_stats = {}
        self.layer_sparsity_trends = defaultdict(list)
        
    def record_activation(self, layer_name: str, activation, timestamp=None):
        """记录激活并计算稀疏度"""
        if timestamp is None:
            timestamp = time.time()
            
        sparsity = self._calculate_sparsity(activation)
        self.sparsity_history.append((layer_name, sparsity, timestamp))
        
        if layer_name not in self.activation_stats:
            self.activation_stats[layer_name] = []
        self.activation_stats[layer_name].append(sparsity)
        self.layer_sparsity_trends[layer_name].append((timestamp, sparsity))
        
        return sparsity
    
    def _calculate_sparsity(self, activation) -> float:
        """计算稀疏度"""
        if isinstance(activation, Tensor):
            data = activation.data
        else:
            data = activation
            
        zero_threshold = 1e-6
        zero_count = np.sum(np.abs(data) < zero_threshold)
        total_elements = np.prod(data.shape)
        
        return zero_count / total_elements if total_elements > 0 else 0.0
    
    def get_layer_sparsity(self, layer_name: str) -> float:
        """获取指定层的平均稀疏度"""
        if layer_name not in self.activation_stats:
            return 0.0
        return np.mean(self.activation_stats[layer_name])
    
    def get_overall_sparsity(self) -> float:
        """获取整体稀疏度"""
        if not self.sparsity_history:
            return 0.0
        return np.mean([sparsity for _, sparsity, _ in self.sparsity_history])
    
    def get_sparsity_trend(self, layer_name: str) -> List[tuple]:
        """获取指定层的稀疏度趋势"""
        return self.layer_sparsity_trends.get(layer_name, [])
    
    def get_most_sparse_layers(self, top_k: int = 5) -> List[tuple]:
        """获取最稀疏的层"""
        layer_sparsities = []
        for layer_name in self.activation_stats:
            avg_sparsity = self.get_layer_sparsity(layer_name)
            layer_sparsities.append((layer_name, avg_sparsity))
        
        return sorted(layer_sparsities, key=lambda x: x[1], reverse=True)[:top_k]
    
    def generate_report(self) -> Dict[str, Any]:
        """生成稀疏度报告"""
        return {
            'overall_sparsity': self.get_overall_sparsity(),
            'layer_sparsities': {layer: self.get_layer_sparsity(layer) 
                               for layer in self.activation_stats},
            'most_sparse_layers': self.get_most_sparse_layers(),
            'total_measurements': len(self.sparsity_history),
            'monitored_layers': len(self.activation_stats)
        }

class CarbonFootprintTracker:
    """碳足迹追踪器"""
    
    def __init__(self, carbon_intensity=0.5, power_draw_watts=200.0):  # kgCO2/kWh, 默认值
        self.carbon_intensity = carbon_intensity
        self.power_draw_watts = power_draw_watts
        self.total_energy_kwh = 0.0
        self.training_start_time = None
        self.inference_energy = 0.0
        self.training_energy = 0.0
        
    def start_training(self):
        """开始训练计时"""
        self.training_start_time = time.time()
        
    def end_training(self):
        """结束训练计时"""
        if self.training_start_time is not None:
            training_time_hours = (time.time() - self.training_start_time) / 3600
            energy_kwh = (self.power_draw_watts / 1000) * training_time_hours
            self.training_energy += energy_kwh
            self.total_energy_kwh += energy_kwh
            self.training_start_time = None
    
    def record_inference(self, flops: int, duration: float):
        """记录推理能耗"""
        # 简化模型：FLOPs转换为能量
        energy_joules = flops * 1e-9  # 假设1GFLOP = 1焦耳
        energy_kwh = energy_joules / 3.6e6  # 焦耳转kWh
        self.inference_energy += energy_kwh
        self.total_energy_kwh += energy_kwh
    
    def estimate_carbon_footprint(self, energy_kwh: float = None) -> float:
        """估计碳足迹"""
        if energy_kwh is None:
            energy_kwh = self.total_energy_kwh
            
        # 碳足迹 = 能量 * 碳强度
        carbon_footprint_kg = energy_kwh * self.carbon_intensity
        return carbon_footprint_kg
    
    def get_training_carbon(self) -> float:
        """获取训练碳足迹"""
        return self.estimate_carbon_footprint(self.training_energy)
    
    def get_inference_carbon(self) -> float:
        """获取推理碳足迹"""
        return self.estimate_carbon_footprint(self.inference_energy)
    
    def get_total_carbon_footprint(self) -> float:
        """获取总碳足迹"""
        return self.estimate_carbon_footprint()
    
    def generate_report(self) -> Dict[str, Any]:
        """生成碳足迹报告"""
        return {
            'total_carbon_kg': self.get_total_carbon_footprint(),
            'training_carbon_kg': self.get_training_carbon(),
            'inference_carbon_kg': self.get_inference_carbon(),
            'total_energy_kwh': self.total_energy_kwh,
            'training_energy_kwh': self.training_energy,
            'inference_energy_kwh': self.inference_energy,
            'carbon_intensity': self.carbon_intensity
        }

class FLOPsCounter:
    """FLOPs计数器"""
    
    @staticmethod
    def estimate_layer_flops(module, input_shape, output_shape) -> int:
        """估计单层FLOPs"""
        if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
            # Linear层
            batch_size = input_shape[0]
            return 2 * module.in_features * module.out_features * batch_size
            
        elif hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
            # Conv2d层
            batch_size, in_c, in_h, in_w = input_shape
            out_c, out_h, out_w = output_shape[1], output_shape[2], output_shape[3]
            
            if hasattr(module, 'kernel_size'):
                k_h, k_w = module.kernel_size
            else:
                k_h, k_w = 3, 3
                
            # 每个输出位置的FLOPs: in_c * k_h * k_w * 2 (乘加)
            flops_per_position = in_c * k_h * k_w * 2
            return flops_per_position * out_c * out_h * out_w * batch_size
            
        elif hasattr(module, '__class__') and module.__class__.__name__ == 'ReLU':
            # ReLU激活 (比较操作)
            return np.prod(input_shape)
            
        else:
            # 默认估计
            return np.prod(input_shape) * 10
    
    @staticmethod
    def estimate_model_flops(model, input_shape) -> int:
        """估计模型总FLOPs"""
        total_flops = 0
        
        # 模拟前向传播计算每层FLOPs
        # 注意：这是简化版本，实际需要更复杂的计算
        current_shape = input_shape
        for module in model.modules():
            if len(list(module.children())) == 0:  # 只处理叶子模块
                # 估计输出形状（简化）
                if hasattr(module, 'out_features'):
                    output_shape = (current_shape[0], module.out_features)
                elif hasattr(module, 'out_channels'):
                    # 简化：假设高度宽度减半
                    output_shape = (
                        current_shape[0], 
                        module.out_channels, 
                        current_shape[2] // 2, 
                        current_shape[3] // 2
                    )
                else:
                    output_shape = current_shape
                    
                flops = FLOPsCounter.estimate_layer_flops(module, current_shape, output_shape)
                total_flops += flops
                current_shape = output_shape
                
        return total_flops