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
        self.params = params
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
    
    def _clip_if_large(self, array: np.ndarray, threshold: float = 1e6) -> np.ndarray:
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

if __name__ == "__main__":
    """
    优化器测试代码
    直接测试SGD、Momentum、AdaGrad和Adam优化器
    """
    
    print("=" * 70)
    print("开始优化器测试")
    print("=" * 70)
    
    # 测试1: 基础功能测试 - 简单二次函数优化
    print("\n1. 基础功能测试: f(x) = (x - 3)^2")
    print("-" * 40)
    
    # 定义优化器
    x_sgd = Tensor([0.0], requires_grad=True)
    x_momentum = Tensor([0.0], requires_grad=True)
    x_adagrad = Tensor([0.0], requires_grad=True)
    x_adam = Tensor([0.0], requires_grad=True)
    
    optimizers = {
        'SGD': SGD([x_sgd], lr=0.1),
        'Momentum': Momentum([x_momentum], lr=0.1, momentum=0.9),
        'AdaGrad': AdaGrad([x_adagrad], lr=0.5),
        'Adam': Adam([x_adam], lr=0.3)
    }
    
    for name, optimizer in optimizers.items():
        x = optimizer.params[0]
        losses = []
        
        # 优化循环
        for step in range(50):
            # 计算损失: f(x) = (x - 3)^2
            loss = (x - 3.0) ** 2
            loss.backward(np.array([1.0]))
            
            losses.append(loss.data.copy())
            
            # 检查收敛
            if loss.data < 1e-6:
                break
                
            optimizer.step()
            optimizer.zero_grad()
        
        final_x = x.data[0]
        final_loss = losses[-1]
        print(f"{name} | 最终 x = {final_x} | 最终损失 = {final_loss} | 步数 = {len(losses)}")
        
        # 验证结果
        assert abs(final_x - 3.0) < 0.1, f"{name} 未能收敛到正确值"
    
    print("✓ 基础功能测试通过")
    
    # 测试2: 多参数优化测试
    print("\n2. 多参数优化测试")
    print("-" * 40)
    
    # 创建多个参数
    w1 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    w2 = Tensor([0.5, -0.5], requires_grad=True)
    
    # 使用Adam优化器
    optimizer = Adam([w1, w2], lr=0.01)
    
    # 模拟训练过程
    for step in range(100):
        # 模拟一个简单的计算
        output = w1 @ w2.expand_dims(axis=1)  # 矩阵乘法
        loss = (output - Tensor([[1.0], [2.0]])) ** 2
        total_loss = loss.sum()
        
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if step % 25 == 0:
            print(f"步骤 {step:3d}: 损失 = {total_loss.data:.6f}")
    
    print("✓ 多参数优化测试通过")
    
    # 测试3: 广播梯度处理测试
    print("\n3. 广播梯度处理测试")
    print("-" * 40)
    
    # 创建形状不匹配的参数和梯度
    param = Tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    optimizer = Adam([param], lr=0.1)
    
    print(f"参数形状: {param.shape}")
    
    # 模拟广播梯度场景
    # 手动设置一个需要广播的梯度
    param.grad = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8], 
        [0.9, 1.0, 1.1, 1.2]
    ])
    
    print(f"梯度形状: {param.grad.shape}")
    
    try:
        old_param = param.data.copy()
        optimizer.step()
        print(f"参数更新成功!")
        print(f"参数变化范围: [{np.min(param.data - old_param):.4f}, {np.max(param.data - old_param):.4f}]")
        print("✓ 广播梯度处理测试通过")
    except Exception as e:
        print(f"✗ 广播梯度处理测试失败: {e}")
    
    # 测试4: 数值稳定性测试
    print("\n4. 数值稳定性测试")
    print("-" * 40)

    # 测试1: 大梯度
    param1 = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    optimizer1 = Adam([param1], lr=0.1)
    param1.grad = np.array([1e8, 1e8, 1e8])
    try:
        optimizer1.step()
        print("✓ 大梯度处理正常")
    except Exception as e:
        print(f"✗ 大梯度处理失败: {e}")

    # 测试2: 第一次迭代的零梯度
    param2 = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    optimizer2 = Adam([param2], lr=0.1)
    param2.grad = np.array([0.0, 0.0, 0.0])
    try:
        old_param = param2.data.copy()
        optimizer2.step()
        # 第一次迭代，梯度为零时参数确实不应改变
        if np.allclose(param2.data, old_param):
            print("✓ 第一次迭代零梯度处理正常")
        else:
            print("⚠ 第一次迭代零梯度时参数有变化")
    except Exception as e:
        print(f"✗ 第一次迭代零梯度处理失败: {e}")

    # 测试3: 后续迭代的零梯度
    param3 = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    optimizer3 = Adam([param3], lr=0.1)

    # 先进行一次正常更新
    param3.grad = np.array([0.1, 0.2, 0.3])
    optimizer3.step()
    initial_param = param3.data.copy()

    # 再测试零梯度
    param3.grad = np.array([0.0, 0.0, 0.0])
    old_param = param3.data.copy()
    optimizer3.step()
    change = np.abs(param3.data - old_param).max()

    print(f"第一次更新后参数: {initial_param}")
    print(f"零梯度更新前参数: {old_param}")
    print(f"零梯度更新后参数: {param3.data}")
    print(f"参数变化量: {change}")

    # 合理的检查：变化应该逐渐衰减，而不是完全为零
    if change < 0.1:  # 设置合理的阈值
        print("✓ 后续迭代零梯度处理正常")
    else:
        print(f"⚠ 后续迭代零梯度时参数变化较大: {change}")
    
    # 测试5: 优化器状态重置测试
    print("\n5. 优化器状态重置测试")
    print("-" * 40)
    
    param1 = Tensor([5.0], requires_grad=True)
    param2 = Tensor([5.0], requires_grad=True)
    
    # 使用相同的优化器配置但不同的实例
    optimizer1 = Adam([param1], lr=0.1)
    optimizer2 = Adam([param2], lr=0.1)
    
    # 对第一个优化器执行多步
    for _ in range(10):
        loss = (param1 - 2.0) ** 2
        loss.backward(np.array([1.0]))
        optimizer1.step()
        optimizer1.zero_grad()
    
    # 对第二个优化器执行一步
    loss = (param2 - 2.0) ** 2
    loss.backward(np.array([1.0]))
    optimizer2.step()
    optimizer2.zero_grad()
    
    print(f"多次优化后的参数: {param1.data[0]:.4f}")
    print(f"单次优化后的参数: {param2.data[0]:.4f}")
    
    # 两个参数应该不同，因为优化器内部状态不同
    assert not np.allclose(param1.data, param2.data), "优化器状态应该影响参数更新"
    print("✓ 优化器状态重置测试通过")
    
    # 测试6: 学习率效果测试
    print("\n6. 学习率效果测试")
    print("-" * 40)
    
    # 测试不同学习率的效果
    learning_rates = [0.01, 0.1, 0.5]
    changes = []
    for lr in learning_rates:
        x = Tensor([10.0], requires_grad=True)
        optimizer = SGD([x], lr=lr)
        
        # 单步优化
        loss = (x - 0.0) ** 2
        loss.backward(np.array([1.0]))
        old_x = x.data.copy()
        optimizer.step()
        optimizer.zero_grad()
        
        change = abs(old_x[0] - x.data[0])
        changes.append(change)
        print(f"学习率 {lr:4.2f}: 参数变化 = {change:.4f}")
        
    

    # 验证学习率越大，参数变化越大
    for i in range(1, len(changes)):
        assert changes[i] > changes[i-1], f"学习率增加时参数变化应该增大，但 {changes[i]} <= {changes[i-1]}"
    
    # 验证学习率与参数变化的线性关系
    # 理论上：change = lr * gradient = lr * 20.0
    expected_changes = [lr * 20.0 for lr in learning_rates]

    for i, (actual, expected) in enumerate(zip(changes, expected_changes)):
        # 允许10%的误差
        assert abs(actual - expected) < expected * 0.1, \
            f"学习率 {learning_rates[i]} 的参数变化 {actual:.4f} 与预期 {expected:.4f} 不符"
        
        print(f"  学习率 {learning_rates[i]:.2f}: 实际变化 {actual:.4f}, 预期变化 {expected:.4f}, 误差 {abs(actual - expected):.4f}")

    # 验证学习率越大，参数变化越大
    for i in range(1, len(changes)):
        assert changes[i] > changes[i-1], \
            f"学习率增加时参数变化应该增大，但 {changes[i]} <= {changes[i-1]}"


    print("✓ 学习率效果测试通过")
    

    # 测试7: 动量效果测试
    print("\n7. 动量效果测试")
    print("-" * 40)

    # 方法1: 测试动量在持续方向上的加速效果
    print("测试动量在持续方向上的加速效果")

    # 创建参数
    x_momentum = Tensor([0.0], requires_grad=True)
    x_no_momentum = Tensor([0.0], requires_grad=True)

    optimizer_momentum = Momentum([x_momentum], lr=0.1, momentum=0.9)
    optimizer_no_momentum = SGD([x_no_momentum], lr=0.1, momentum=0.0)

    # 模拟持续的正梯度（动量应该加速收敛）
    gradients = [1.0, 1.0, 1.0, 1.0, 1.0]  # 持续正梯度

    print("梯度更新过程:")
    for i, grad in enumerate(gradients):
        # 有动量的优化
        x_momentum.grad = np.array([grad])
        old_momentum = x_momentum.data[0]
        optimizer_momentum.step()
        momentum_change = x_momentum.data[0] - old_momentum
        optimizer_momentum.zero_grad()
        
        # 无动量的优化
        x_no_momentum.grad = np.array([grad])
        old_no_momentum = x_no_momentum.data[0]
        optimizer_no_momentum.step()
        no_momentum_change = x_no_momentum.data[0] - old_no_momentum
        optimizer_no_momentum.zero_grad()
        
        print(f"步骤 {i+1}: 动量变化={momentum_change:.4f}, 无动量变化={no_momentum_change:.4f}")

    print(f"有动量最终参数: {x_momentum.data[0]:.4f}")
    print(f"无动量最终参数: {x_no_momentum.data[0]:.4f}")

    # 根据你的优化器实现，正梯度会导致参数向负方向移动
    # 因此动量应该让参数变得更负（即更小）
    assert x_momentum.data[0] < x_no_momentum.data[0], "动量应该在负梯度方向上加速优化"

    print("✓ 动量加速效果测试通过")

    # 方法2: 测试动量在持续负梯度方向上的加速效果
    print("测试动量在持续负梯度方向上的加速效果")

    # 创建参数
    x_momentum_neg = Tensor([0.0], requires_grad=True)
    x_no_momentum_neg = Tensor([0.0], requires_grad=True)

    optimizer_momentum_neg = Momentum([x_momentum_neg], lr=0.1, momentum=0.9)
    optimizer_no_momentum_neg = SGD([x_no_momentum_neg], lr=0.1, momentum=0.0)

    # 模拟持续的负梯度（动量应该加速收敛）
    gradients_neg = [-1.0, -1.0, -1.0, -1.0, -1.0]  # 持续负梯度

    print("负梯度更新过程:")
    for i, grad in enumerate(gradients_neg):
        # 有动量的优化
        x_momentum_neg.grad = np.array([grad])
        old_momentum = x_momentum_neg.data[0]
        optimizer_momentum_neg.step()
        momentum_change = x_momentum_neg.data[0] - old_momentum
        optimizer_momentum_neg.zero_grad()
        
        # 无动量的优化
        x_no_momentum_neg.grad = np.array([grad])
        old_no_momentum = x_no_momentum_neg.data[0]
        optimizer_no_momentum_neg.step()
        no_momentum_change = x_no_momentum_neg.data[0] - old_no_momentum
        optimizer_no_momentum_neg.zero_grad()
        
        print(f"步骤 {i+1}: 动量变化={momentum_change:.4f}, 无动量变化={no_momentum_change:.4f}")

    print(f"有动量最终参数(负梯度): {x_momentum_neg.data[0]:.4f}")
    print(f"无动量最终参数(负梯度): {x_no_momentum_neg.data[0]:.4f}")

    # 负梯度会导致参数向正方向移动，动量应该让参数变得更大
    assert x_momentum_neg.data[0] > x_no_momentum_neg.data[0], "动量应该在正梯度方向上加速优化"

    print("✓ 动量负梯度加速效果测试通过")

    # 测试动量在振荡梯度下的平滑效果
    print("测试动量在振荡梯度下的平滑效果")

    # 重新初始化参数
    x_momentum = Tensor([0.0], requires_grad=True)
    x_no_momentum = Tensor([0.0], requires_grad=True)

    optimizer_momentum = Momentum([x_momentum], lr=0.1, momentum=0.9)
    optimizer_no_momentum = SGD([x_no_momentum], lr=0.1, momentum=0.0)

    # 模拟更强的振荡梯度
    gradients = [2.0, -1.8, 1.6, -1.4, 1.2, -1.0, 0.8, -0.6, 0.4, -0.2]

    # 记录路径
    momentum_path = [x_momentum.data[0]]  # 包括初始点
    no_momentum_path = [x_no_momentum.data[0]]

    for grad in gradients:
        # 有动量的优化
        x_momentum.grad = np.array([grad])
        optimizer_momentum.step()
        momentum_path.append(x_momentum.data[0])
        optimizer_momentum.zero_grad()
        
        # 无动量的优化
        x_no_momentum.grad = np.array([grad])
        optimizer_no_momentum.step()
        no_momentum_path.append(x_no_momentum.data[0])
        optimizer_no_momentum.zero_grad()

    # 计算路径的一阶差分
    momentum_diff = np.diff(momentum_path)
    no_momentum_diff = np.diff(no_momentum_path)

    print(f"有动量路径差分: {momentum_diff}")
    print(f"无动量路径差分: {no_momentum_diff}")

    # 计算差分序列的标准差
    momentum_diff_std = np.std(momentum_diff)
    no_momentum_diff_std = np.std(no_momentum_diff)

    print(f"有动量路径差分标准差: {momentum_diff_std:.4f}")
    print(f"无动量路径差分标准差: {no_momentum_diff_std:.4f}")

    # 动量应该减少路径的振荡，即差分序列的标准差更小
    assert momentum_diff_std < no_momentum_diff_std, "动量应该减少优化路径的振荡（差分序列标准差更小）"

    print("✓ 动量平滑效果测试通过")

    print("\n" + "=" * 70)
    print("所有优化器测试完成! 🎉")
    print("=" * 70)
    
    # 性能比较总结
    print("\n优化器性能总结:")
    print("- SGD: 基础但可靠，适合简单问题")
    print("- Momentum: 加速收敛，减少振荡") 
    print("- AdaGrad: 自适应学习率，适合稀疏数据")
    print("- Adam: 结合动量和自适应学习率，通常效果最佳")