import numpy as np
import random
from typing import List, Dict, Any, Tuple
import sys
sys.path.append("..")
from nn.model import Module

import numpy as np
import random
import time
from typing import List, Dict, Any, Tuple, Optional, Callable
from abc import ABC, abstractmethod
from nn.model import Module

class EnergyAwareNAS:
    """能量感知的神经架构搜索 - 将能量作为NAS的奖励"""
    
    def __init__(self, search_space: Dict[str, Any], 
                 energy_weight: float = 0.3, 
                 accuracy_weight: float = 0.7,
                 latency_weight: float = 0.0,
                 target_device: str = "cpu"):
        self.search_space = search_space
        self.energy_weight = energy_weight
        self.accuracy_weight = accuracy_weight
        self.latency_weight = latency_weight
        self.target_device = target_device
        
        self.architectures = []
        self.performance_history = []
        self.best_architecture = None
        self.best_score = -float('inf')
        
    def evaluate_architecture(self, 
                            architecture_config: Dict[str, Any], 
                            train_func: Callable,
                            input_shape: Tuple[int, ...],
                            energy_calculator: Optional[Callable] = None) -> Tuple[float, Dict[str, float]]:
        """评估架构的性能和能量效率"""
        try:
            # 创建模型
            model = self._create_model_from_config(architecture_config)
            
            # 训练并评估准确率
            accuracy = train_func(model)
            
            # 评估能量效率
            energy_efficiency, metrics = self._evaluate_energy_efficiency(
                model, input_shape, energy_calculator
            )
            
            # 评估延迟（如果可用）
            latency_score = self._evaluate_latency(model, input_shape)
            
            # 计算综合分数
            score = (
                self.accuracy_weight * accuracy + 
                self.energy_weight * energy_efficiency +
                self.latency_weight * latency_score
            )
            
            metrics.update({
                'score': score,
                'accuracy': accuracy,
                'energy_efficiency': energy_efficiency,
                'latency_score': latency_score
            })
            
            return score, metrics
            
        except Exception as e:
            print(f"Architecture evaluation failed: {e}")
            return 0.0, {'error': str(e)}
    
    def _evaluate_energy_efficiency(self, 
                                  model: Module, 
                                  input_shape: Tuple[int, ...],
                                  energy_calculator: Optional[Callable] = None) -> Tuple[float, Dict[str, float]]:
        """评估架构的能量效率"""
        if energy_calculator is not None:
            # 使用提供的能量计算器
            energy_metrics = energy_calculator(model, input_shape)
            energy_efficiency = energy_metrics.get('efficiency', 0.0)
            return energy_efficiency, energy_metrics
        
        # 默认能量效率评估
        total_params = sum(np.prod(p.shape) for p in model.parameters())
        total_flops = self._estimate_model_flops(model, input_shape)
        
        # 能量效率分数 (越高越好)
        # 使用FLOPs和参数数量的倒数，因为越小越好
        params_efficiency = 1.0 / (1.0 + total_params / 1e6)  # 归一化
        flops_efficiency = 1.0 / (1.0 + total_flops / 1e9)   # 归一化
        
        energy_efficiency = 0.7 * flops_efficiency + 0.3 * params_efficiency
        
        metrics = {
            'total_params': total_params,
            'total_flops': total_flops,
            'params_efficiency': params_efficiency,
            'flops_efficiency': flops_efficiency
        }
        
        return energy_efficiency, metrics
    
    def _estimate_model_flops(self, model: Module, input_shape: Tuple[int, ...]) -> int:
        """估计模型总FLOPs"""
        total_flops = 0
        current_shape = input_shape
        
        # 简化版的FLOPs估计
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 只处理叶子模块
                if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                    # Linear层
                    batch_size = current_shape[0]
                    flops = 2 * module.in_features * module.out_features * batch_size
                    current_shape = (batch_size, module.out_features)
                    
                elif hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
                    # Conv2d层
                    batch_size, in_c, in_h, in_w = current_shape
                    if hasattr(module, 'kernel_size'):
                        k_h, k_w = module.kernel_size
                    else:
                        k_h, k_w = 3, 3
                    
                    # 简化输出形状计算
                    out_h = in_h // 2  # 假设步长为2
                    out_w = in_w // 2
                    out_c = module.out_channels
                    
                    flops = 2 * in_c * k_h * k_w * out_c * out_h * out_w * batch_size
                    current_shape = (batch_size, out_c, out_h, out_w)
                    
                else:
                    flops = 1000  # 默认值
                    
                total_flops += flops
                
        return total_flops
    
    def _evaluate_latency(self, model: Module, input_shape: Tuple[int, ...]) -> float:
        """评估模型延迟（简化版本）"""
        # 简化延迟评估：基于参数数量和FLOPs
        total_params = sum(np.prod(p.shape) for p in model.parameters())
        total_flops = self._estimate_model_flops(model, input_shape)
        
        # 延迟分数 (越高越好)
        latency_score = 1.0 / (1.0 + (total_params / 1e6 + total_flops / 1e9))
        return latency_score
    
    def _create_model_from_config(self, config: Dict[str, Any]) -> Module:
        """根据配置创建模型"""
        # 这里需要根据你的模型结构来实现
        # 简化版本，返回一个基本模型
        from ..nn.layer import Linear, Conv2d, ReLU
        from ..nn.model import Sequential
        
        layers = []
        for i, layer_config in enumerate(config.get('layers', [])):
            layer_type = layer_config['type']
            if layer_type == 'Linear':
                layer = Linear(layer_config['in_features'], 
                              layer_config['out_features'])
            elif layer_type == 'Conv2d':
                layer = Conv2d(layer_config['in_channels'], 
                              layer_config['out_channels'],
                              kernel_size=layer_config.get('kernel_size', 3))
            elif layer_type == 'ReLU':
                layer = ReLU()
            else:
                continue
            layers.append(layer)
            
        return Sequential(*layers)
    
    def random_search(self, 
                     num_iterations: int, 
                     train_func: Callable,
                     input_shape: Tuple[int, ...],
                     energy_calculator: Optional[Callable] = None) -> Dict[str, Any]:
        """随机搜索最佳架构"""
        
        for i in range(num_iterations):
            # 随机采样架构
            architecture_config = self._sample_architecture()
            
            # 评估架构
            score, metrics = self.evaluate_architecture(
                architecture_config, train_func, input_shape, energy_calculator
            )
            
            # 记录结果
            result = {
                'config': architecture_config,
                'score': score,
                'metrics': metrics,
                'iteration': i,
                'timestamp': time.time()
            }
            self.architectures.append(result)
            self.performance_history.append(score)
            
            # 更新最佳架构
            if score > self.best_score:
                self.best_score = score
                self.best_architecture = architecture_config
                self.best_metrics = metrics
                
            print(f"Iteration {i+1}/{num_iterations}: "
                  f"Score={score:.4f}, "
                  f"Accuracy={metrics.get('accuracy', 0):.4f}, "
                  f"EnergyEff={metrics.get('energy_efficiency', 0):.4f}")
        
        return {
            'best_architecture': self.best_architecture,
            'best_score': self.best_score,
            'best_metrics': self.best_metrics,
            'all_architectures': self.architectures,
            'performance_history': self.performance_history
        }
    
    def _sample_architecture(self) -> Dict[str, Any]:
        """从搜索空间中随机采样一个架构"""
        # 简化版的随机采样
        num_layers = random.randint(3, 8)
        layers = []
        
        in_features = 784  # MNIST输入
        out_features = 10  # MNIST输出
        
        for i in range(num_layers - 1):  # 最后一层是输出层
            layer_type = random.choice(['Linear', 'ReLU'])
            
            if layer_type == 'Linear':
                hidden_size = random.choice([64, 128, 256, 512])
                layers.append({
                    'type': 'Linear',
                    'in_features': in_features,
                    'out_features': hidden_size
                })
                in_features = hidden_size
            else:
                layers.append({'type': 'ReLU'})
        
        # 输出层
        layers.append({
            'type': 'Linear',
            'in_features': in_features,
            'out_features': out_features
        })
        
        return {'layers': layers}
    
    def get_search_report(self) -> Dict[str, Any]:
        """获取搜索报告"""
        if not self.architectures:
            return {}
            
        scores = [arch['score'] for arch in self.architectures]
        accuracies = [arch['metrics'].get('accuracy', 0) for arch in self.architectures]
        energy_effs = [arch['metrics'].get('energy_efficiency', 0) for arch in self.architectures]
        
        return {
            'total_evaluations': len(self.architectures),
            'best_score': self.best_score,
            'average_score': np.mean(scores),
            'std_score': np.std(scores),
            'average_accuracy': np.mean(accuracies),
            'average_energy_efficiency': np.mean(energy_effs),
            'search_config': {
                'energy_weight': self.energy_weight,
                'accuracy_weight': self.accuracy_weight,
                'latency_weight': self.latency_weight,
                'target_device': self.target_device
            }
        }

class EvolutionaryEnergyNAS(EnergyAwareNAS):
    """进化算法能量感知NAS"""
    
    def __init__(self, search_space: Dict[str, Any], 
                 population_size: int = 20,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 **kwargs):
        super().__init__(search_space, **kwargs)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.generation = 0
        
    def initialize_population(self):
        """初始化种群"""
        self.population = []
        for _ in range(self.population_size):
            architecture = self._sample_architecture()
            self.population.append(architecture)
    
    def evolutionary_search(self, 
                          num_generations: int,
                          train_func: Callable,
                          input_shape: Tuple[int, ...],
                          energy_calculator: Optional[Callable] = None) -> Dict[str, Any]:
        """进化搜索"""
        
        # 初始化种群
        if not self.population:
            self.initialize_population()
        
        for generation in range(num_generations):
            self.generation = generation
            
            # 评估当前种群
            evaluated_population = []
            for arch in self.population:
                score, metrics = self.evaluate_architecture(
                    arch, train_func, input_shape, energy_calculator
                )
                evaluated_population.append((arch, score, metrics))
            
            # 排序并选择
            evaluated_population.sort(key=lambda x: x[1], reverse=True)
            selected = self._select(evaluated_population)
            
            # 交叉和变异
            new_population = self._crossover_and_mutate(selected)
            self.population = new_population
            
            # 记录最佳个体
            best_arch, best_score, best_metrics = evaluated_population[0]
            if best_score > self.best_score:
                self.best_score = best_score
                self.best_architecture = best_arch
                self.best_metrics = best_metrics
            
            print(f"Generation {generation+1}/{num_generations}: "
                  f"Best Score={best_score:.4f}, "
                  f"Avg Score={np.mean([x[1] for x in evaluated_population]):.4f}")
        
        return {
            'best_architecture': self.best_architecture,
            'best_score': self.best_score,
            'best_metrics': self.best_metrics,
            'final_population': self.population,
            'total_generations': num_generations
        }
    
    def _select(self, evaluated_population: List[tuple], 
               selection_ratio: float = 0.5) -> List[Dict[str, Any]]:
        """选择操作"""
        num_selected = int(len(evaluated_population) * selection_ratio)
        selected = [arch for arch, _, _ in evaluated_population[:num_selected]]
        
        # 轮盘赌选择补充
        scores = np.array([score for _, score, _ in evaluated_population])
        probabilities = scores / np.sum(scores)
        
        additional_indices = np.random.choice(
            len(evaluated_population), 
            size=len(evaluated_population) - num_selected,
            p=probabilities
        )
        
        for idx in additional_indices:
            selected.append(evaluated_population[idx][0])
            
        return selected
    
    def _crossover_and_mutate(self, selected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """交叉和变异操作"""
        new_population = []
        
        # 保留精英
        new_population.extend(selected[:2])
        
        # 生成新个体
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate and len(selected) >= 2:
                # 交叉
                parent1, parent2 = random.sample(selected, 2)
                child = self._crossover(parent1, parent2)
            else:
                # 直接复制
                child = random.choice(selected).copy()
            
            # 变异
            if random.random() < self.mutation_rate:
                child = self._mutate(child)
                
            new_population.append(child)
        
        return new_population
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """交叉操作"""
        # 简化交叉：随机选择父代的层
        child_layers = []
        max_layers = max(len(parent1['layers']), len(parent2['layers']))
        
        for i in range(max_layers):
            if i < len(parent1['layers']) and i < len(parent2['layers']):
                # 随机选择父代
                chosen_parent = random.choice([parent1, parent2])
                child_layers.append(chosen_parent['layers'][i])
            elif i < len(parent1['layers']):
                child_layers.append(parent1['layers'][i])
            else:
                child_layers.append(parent2['layers'][i])
                
        return {'layers': child_layers}
    
    def _mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """变异操作"""
        mutated = individual.copy()
        layers = mutated['layers'].copy()
        
        mutation_type = random.choice(['add_layer', 'remove_layer', 'modify_layer'])
        
        if mutation_type == 'add_layer' and len(layers) < 10:
            # 添加层
            new_layer = {
                'type': 'Linear',
                'in_features': layers[-1]['out_features'] if layers else 784,
                'out_features': random.choice([64, 128, 256])
            }
            layers.append(new_layer)
            
        elif mutation_type == 'remove_layer' and len(layers) > 2:
            # 移除层（不能移除输入和输出层）
            remove_idx = random.randint(1, len(layers) - 2)
            layers.pop(remove_idx)
            
        elif mutation_type == 'modify_layer' and len(layers) > 0:
            # 修改层
            modify_idx = random.randint(0, len(layers) - 1)
            if layers[modify_idx]['type'] == 'Linear':
                layers[modify_idx]['out_features'] = random.choice([64, 128, 256, 512])
        
        mutated['layers'] = layers
        return mutated