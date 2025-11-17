import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nn import Linear, Dropout, BatchNorm1d, MaxPool2d, Conv2d, Flatten, BatchNorm2d
from core import Tensor



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


print("\n=== CNN层测试 ===")

# 测试Conv2d
print("1. Conv2d测试")
conv = Conv2d(1, 3, kernel_size=3, padding=1)
x = Tensor(np.random.randn(2, 1, 5, 5) * 0.1, requires_grad=True)
y = conv(x)
print(f"输入形状: {x.shape}")
print(f"输出形状: {y.shape}")

# 测试梯度
loss = y.sum()
loss.backward()
print(f"权重梯度形状: {conv.weight.grad.shape}")
if conv.bias_param:
    print(f"偏置梯度形状: {conv.bias_param.grad.shape}")
print("✓ Conv2d测试通过")

# 测试MaxPool2d
print("\n2. MaxPool2d测试")
pool = MaxPool2d(2)
x_pool = Tensor(np.random.randn(2, 3, 4, 4) * 0.1, requires_grad=True)
y_pool = pool(x_pool)
print(f"输入形状: {x_pool.shape}")
print(f"输出形状: {y_pool.shape}")

loss_pool = y_pool.sum()
loss_pool.backward()
print("✓ MaxPool2d测试通过")

# 测试Flatten
print("\n3. Flatten测试")
flatten = Flatten()
x_flat = Tensor(np.random.randn(2, 3, 4, 4) * 0.1, requires_grad=True)
y_flat = flatten(x_flat)
print(f"输入形状: {x_flat.shape}")
print(f"输出形状: {y_flat.shape}")

loss_flat = y_flat.sum()
loss_flat.backward()
print("✓ Flatten测试通过")

print("=== 测试CNN层 ===")
    
# 测试Conv2d
print("1. Conv2d测试")
conv = Conv2d(1, 3, kernel_size=3, padding=1)
x = Tensor(np.random.randn(2, 1, 5, 5) * 0.1, requires_grad=True)
y = conv(x)
print(f"输入形状: {x.shape}")
print(f"输出形状: {y.shape}")

# 测试梯度
loss = y.sum()
loss.backward()
print(f"权重梯度形状: {conv.weight.grad.shape}")
if conv.bias_param:
    print(f"偏置梯度形状: {conv.bias_param.grad.shape}")
print("✓ Conv2d测试通过")

# 测试MaxPool2d
print("\n2. MaxPool2d测试")
pool = MaxPool2d(2)
x_pool = Tensor(np.random.randn(2, 3, 4, 4) * 0.1, requires_grad=True)
y_pool = pool(x_pool)
print(f"输入形状: {x_pool.shape}")
print(f"输出形状: {y_pool.shape}")

loss_pool = y_pool.sum()
loss_pool.backward()
print("✓ MaxPool2d测试通过")

# 测试Flatten
print("\n3. Flatten测试")
flatten = Flatten()
x_flat = Tensor(np.random.randn(2, 3, 4, 4) * 0.1, requires_grad=True)
y_flat = flatten(x_flat)
print(f"输入形状: {x_flat.shape}")
print(f"输出形状: {y_flat.shape}")

loss_flat = y_flat.sum()
loss_flat.backward()
print("✓ Flatten测试通过")