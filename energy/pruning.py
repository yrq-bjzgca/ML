import numpy as np
from typing import Dict, List, Tuple, Any
import sys
sys.path.append("..")
from core.tensor import Tensor
import time

class EnergyAwarePruner:
    """能量感知剪枝基类"""
    
    def __init__(self, sparsity_target=0.5, energy_aware=True, 
                 min_sparsity=0.1, max_sparsity=0.9):
        self.sparsity_target = sparsity_target
        self.energy_aware = energy_aware
        self.min_sparsity = min_sparsity
        self.max_sparsity = max_sparsity
        self.pruning_masks = {}
        self.pruning_history = []
        
    def compute_mask(self, weight: Tensor, energy_importance: float = 1.0) -> np.ndarray:
        """计算剪枝掩码 - 子类需要实现"""
        raise NotImplementedError("Subclasses must implement compute_mask")
        
    def prune_model(self, model, energy_monitor=None) -> Dict[str, np.ndarray]:
        """剪枝整个模型"""
        masks = {}
        total_pruned = 0
        total_params = 0
        
        parameters = self._get_prunable_parameters(model)
        
        for param_name, param in parameters:
            # 计算能量重要性
            energy_importance = 1.0
            if energy_monitor and self.energy_aware:
                energy_importance = self._compute_energy_importance(param_name, param, energy_monitor)
            
            # 调整稀疏度目标基于能量重要性
            adjusted_sparsity = self._adjust_sparsity_by_energy(
                self.sparsity_target, energy_importance
            )
            
            # 计算剪枝掩码
            mask = self.compute_mask(param, adjusted_sparsity)
            masks[param_name] = mask
            
            # 应用剪枝
            param.data *= mask
            
            # 统计信息
            pruned_count = np.sum(mask == 0)
            total_count = np.prod(mask.shape)
            total_pruned += pruned_count
            total_params += total_count
            
            # 记录剪枝历史
            self.pruning_history.append({
                'param_name': param_name,
                'sparsity': pruned_count / total_count,
                'energy_importance': energy_importance,
                'adjusted_sparsity': adjusted_sparsity,
                'timestamp': time.time()
            })
                
        self.pruning_masks = masks
        
        # 计算总体稀疏度
        overall_sparsity = total_pruned / total_params if total_params > 0 else 0
        
        print(f"Pruning completed: {overall_sparsity:.2%} parameters pruned")
        return masks
    
    # def _get_prunable_parameters(self, model) -> List[Tuple[str, Tensor]]:
    #     """获取可剪枝的参数"""
    #     parameters = []
        
    #     for name, module in model.named_modules():
    #         if hasattr(module, 'weight') and module.weight is not None:
    #             # 只剪枝2D或4D的权重（Linear和Conv2d）
    #             if len(module.weight.shape) in [2, 4]:
    #                 parameters.append((f"{name}.weight", module.weight))
        
    #     return parameters
    
    def _get_prunable_parameters(self, model) -> List[Tuple[str, Tensor]]:
        """获取可剪枝的参数"""
        parameters = []
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                # 只剪枝2D或4D的权重（Linear和Conv2d）
                if len(module.weight.shape) in [2, 4]:
                    # 对于Linear层，返回转置视图以便与计算逻辑一致
                    if hasattr(module, 'in_features'):  # 是Linear层
                        # 返回权重转置视图，形状为 (in_features, out_features)
                        weight_transpose = module.weight.transpose()
                        parameters.append((f"{name}.weight_T", weight_transpose))
                    else:
                        parameters.append((f"{name}.weight", module.weight))
        
        return parameters
    
    def _compute_energy_importance(self, param_name: str, param: Tensor, 
                                 energy_monitor) -> float:
        """计算基于能量的重要性分数"""
        # 基于能量消耗的重要性：能量消耗越高的层，剪枝时越谨慎
        module_name = param_name.rsplit('.', 1)[0]  # 提取模块名称
        
        for stats_name, stats in energy_monitor.activation_stats.items():
            if module_name in stats_name or stats_name in module_name:
                energy = stats.get('energy', 0)
                # 归一化到 [0.5, 2.0] 范围
                # importance = 1.0 + (1.0 / (1.0 + np.exp(-energy / 1e6)))  # sigmoid归一化
                
                # 将能量映射到 [0.9, 1.1] 范围，1.0为中性
                # 高能量层略微减少剪枝，低能量层略微增加
                importance = 1.0 + np.tanh(energy / 1e7) * 0.1
                return min(max(importance, 0.9), 1.1)  # 限制在合理范围内
        
        return 1.0
    
    def _adjust_sparsity_by_energy(self, base_sparsity: float, 
                                 energy_importance: float) -> float:
        """根据能量重要性调整稀疏度"""
        # 能量重要性高的层，减少剪枝强度
        adjusted = base_sparsity / energy_importance
        return min(max(adjusted, self.min_sparsity), self.max_sparsity)
    
    def get_pruning_report(self) -> Dict[str, Any]:
        """获取剪枝报告"""
        if not self.pruning_history:
            return {}
            
        total_sparsity = np.mean([item['sparsity'] for item in self.pruning_history])
        avg_energy_importance = np.mean([item['energy_importance'] for item in self.pruning_history])
        
        return {
            'total_parameters_pruned': len(self.pruning_masks),
            'average_sparsity': total_sparsity,
            'average_energy_importance': avg_energy_importance,
            'pruning_history': self.pruning_history
        }


class MagnitudeEnergyPruner(EnergyAwarePruner):
    """基于幅值和能量的剪枝器 - 精确控制版本"""
    
    def compute_mask(self, weight: Tensor, target_sparsity: float = 0.5) -> np.ndarray:
        """精确控制稀疏度的剪枝（直接按数量剪枝）"""
        weight_data = weight.data
        
        # ===== DEBUG 打印 =====
        # print(f"DEBUG: Weight shape = {weight_data.shape}, target_sparsity = {target_sparsity}")
        # print(f"DEBUG: Weight min abs = {np.min(np.abs(weight_data)):.6f}, max abs = {np.max(np.abs(weight_data)):.6f}")
        # =====================
        
        total_elements = np.prod(weight_data.shape)
        num_prune = int(total_elements * target_sparsity)
        
        if num_prune == 0:
            # print("DEBUG: No pruning (num_prune=0)")
            return np.ones_like(weight_data, dtype=np.float32)
        if num_prune >= total_elements:
            # print("DEBUG: Prune all")
            return np.zeros_like(weight_data, dtype=np.float32)
        
        # 计算幅值并直接找出最小的 num_prune 个
        abs_weights = np.abs(weight_data).flatten()
        
        # 找到第 num_prune 小的值作为阈值
        # np.partition 将前 num_prune 小的值放在前面
        threshold = np.partition(abs_weights, num_prune - 1)[num_prune - 1]
        
        # print(f"DEBUG: Total elements = {total_elements}, num_prune = {num_prune}")
        # print(f"DEBUG: Threshold value = {threshold:.8f}")
        
        # 创建掩码：保留幅值 > 阈值的（严格大于）
        mask = (abs_weights > threshold).astype(np.float32)
        
        # 精确调整：如果实际剪枝数量不足，补充剪枝
        actual_pruned = np.sum(mask == 0)
        if actual_pruned < num_prune:
            # 需要额外剪枝 (num_prune - actual_pruned) 个
            remaining_indices = np.where(mask == 1)[0]
            # 从剩余的权重中找出最小的那些
            extra_prune_count = num_prune - actual_pruned
            if len(remaining_indices) > extra_prune_count:
                extra_indices = remaining_indices[:extra_prune_count]
                mask[extra_indices] = 0
        
        final_mask = mask.reshape(weight_data.shape)
        
        # ===== DEBUG 验证 =====
        final_sparsity = np.sum(final_mask == 0) / total_elements
        # print(f"DEBUG: Final sparsity = {final_sparsity:.3%}")
        # ======================
        
        return final_mask


class GradientEnergyPruner(EnergyAwarePruner):
    """基于梯度的剪枝器 - 修正版本"""
    
    def __init__(self, sparsity_target=0.5, energy_aware=True, gradient_power=1.0):
        super().__init__(sparsity_target, energy_aware)
        self.gradient_power = gradient_power
        
    def compute_mask(self, weight: Tensor, target_sparsity: float = 0.5) -> np.ndarray:
        """精确控制：剪枝梯度最大的权重（因为梯度大说明对损失影响大）"""
        if weight.grad is None:
            # 无梯度时回退到幅值剪枝
            return MagnitudeEnergyPruner(energy_aware=False).compute_mask(weight, target_sparsity)
        
        weight_data = weight.data
        grad_data = weight.grad
        
        # 关键修正：剪枝重要性 = |grad|^power，与权重大小无关
        # 梯度大的参数对损失影响大，应该保留；梯度小的可以剪枝
        importance = np.abs(grad_data) ** self.gradient_power
        
        # 使用精确剪枝方法
        total_elements = np.prod(importance.shape)
        num_prune = int(total_elements * target_sparsity)
        
        if num_prune == 0:
            return np.ones(importance.shape, dtype=np.float32)
        if num_prune >= total_elements:
            return np.zeros(importance.shape, dtype=np.float32)
        
        # 保留重要性高的（保留梯度大的）
        flat_importance = importance.flatten()
        
        # 找到第 num_prune 小的重要性值
        threshold = np.partition(flat_importance, num_prune - 1)[num_prune - 1]
        
        # 重要性 >= 阈值的保留
        mask = (flat_importance >= threshold).astype(np.float32)
        
        # 精确调整：确保剪枝数量精确
        actual_pruned = np.sum(mask == 0)
        if actual_pruned < num_prune:
            # 补充剪枝
            remaining_indices = np.where(mask == 1)[0]
            extra_needed = num_prune - actual_pruned
            mask[remaining_indices[:extra_needed]] = 0
        
        return mask.reshape(importance.shape)

class StructuredEnergyPruner(EnergyAwarePruner):
    """结构化能量感知剪枝器"""
    
    def __init__(self, sparsity_target=0.5, energy_aware=True, 
                 structure_type='channel'):
        super().__init__(sparsity_target, energy_aware)
        self.structure_type = structure_type  # 'channel', 'filter', 'row', 'column'
        
    def compute_mask(self, weight: Tensor, target_sparsity: float = 0.5) -> np.ndarray:
        """结构化剪枝"""
        weight_data = weight.data
        
        if len(weight_data.shape) == 4 and self.structure_type == 'channel':
            # 卷积层的通道剪枝
            return self._channel_wise_pruning(weight_data, target_sparsity)
        elif len(weight_data.shape) == 2 and self.structure_type == 'column':
            # 全连接层的列剪枝
            return self._column_wise_pruning(weight_data, target_sparsity)
        else:
            # 回退到非结构化剪枝
            pruner = MagnitudeEnergyPruner(target_sparsity)
            return pruner.compute_mask(weight, target_sparsity)
    
    def _channel_wise_pruning(self, weight_data: np.ndarray, 
                            target_sparsity: float) -> np.ndarray:
        """通道级剪枝（针对卷积层）"""
        # 计算每个通道的重要性（L2范数）
        channel_importance = np.linalg.norm(weight_data, axis=(1, 2, 3))
        
        # 计算阈值
        threshold = np.percentile(channel_importance, target_sparsity * 100)
        
        # 创建通道掩码
        channel_mask = (channel_importance > threshold).astype(np.float32)
        
        # 扩展到权重形状
        mask = np.ones_like(weight_data)
        for i in range(len(channel_mask)):
            if channel_mask[i] == 0:
                mask[i, :, :, :] = 0
                
        return mask
    
    def _column_wise_pruning(self, weight_data: np.ndarray, 
                           target_sparsity: float) -> np.ndarray:
        """列级剪枝（针对全连接层）"""
        # 计算每列的重要性（L2范数）
        column_importance = np.linalg.norm(weight_data, axis=0)
        
        # 计算阈值
        threshold = np.percentile(column_importance, target_sparsity * 100)
        
        # 创建列掩码
        column_mask = (column_importance > threshold).astype(np.float32)
        
        # 扩展到权重形状
        mask = np.ones_like(weight_data)
        for i in range(len(column_mask)):
            if column_mask[i] == 0:
                mask[:, i] = 0
                
        return mask





class ProgressiveEnergyPruner:
    """渐进式能量感知剪枝器"""
    
    def __init__(self, initial_sparsity=0.1, final_sparsity=0.8, 
                 steps=10, pruner_class=MagnitudeEnergyPruner):
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.steps = steps
        self.pruner_class = pruner_class
        self.current_step = 0
        self.pruners = []
        
    def prune_step(self, model, energy_monitor=None):
        """执行一步渐进式剪枝"""
        if self.current_step >= self.steps:
            print("Progressive pruning completed")
            return None
            
        # 计算当前步的目标稀疏度
        current_sparsity = self.initial_sparsity + (
            self.final_sparsity - self.initial_sparsity
        ) * (self.current_step / (self.steps - 1))
        
        # 创建剪枝器并执行剪枝
        pruner = self.pruner_class(sparsity_target=current_sparsity)
        masks = pruner.prune_model(model, energy_monitor)
        
        self.pruners.append(pruner)
        self.current_step += 1
        
        print(f"Progressive pruning step {self.current_step}/{self.steps}, "
              f"sparsity: {current_sparsity:.2%}")
        
        return masks
    
    def get_progress_report(self) -> Dict[str, Any]:
        """获取渐进式剪枝进度报告"""
        return {
            'current_step': self.current_step,
            'total_steps': self.steps,
            'current_sparsity': self.initial_sparsity + (
                self.final_sparsity - self.initial_sparsity
            ) * (self.current_step / (self.steps - 1)) if self.steps > 1 else self.initial_sparsity,
            'completed': self.current_step >= self.steps
        }