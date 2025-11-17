
import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from energy.regularize import (
    L2Regularizer, 
    FLOPsCalculator,
    FLOPsRegularizer,
    EnergyAwareRegularizer,
    CombinedRegularizer
)
from nn import Sequential, Linear, ReLU, Conv2d
from core.tensor import Tensor

# 辅助函数：检查梯度是否近似相等
def check_grad(computed, expected, eps=1e-5):
    assert np.allclose(computed, expected, atol=eps), \
        f"梯度不匹配: 计算值={computed}，期望值={expected}"

print("===== 开始 1. 基础运算测试（+、-、*、/） =====")
# 测试加法
a = Tensor([2.0, 3.0], requires_grad=True)
b = Tensor([4.0, 5.0], requires_grad=True)
c = a + b
c.backward(np.array([1.0, 1.0]))
check_grad(a.grad, [1.0, 1.0])
check_grad(b.grad, [1.0, 1.0])
a.zero_grad()
b.zero_grad()

# 测试减法
c = a - b
c.backward(np.array([1.0, 1.0]))
check_grad(a.grad, [1.0, 1.0])
check_grad(b.grad, [-1.0, -1.0])
a.zero_grad()
b.zero_grad()

# 测试乘法
c = a * b
c.backward(np.array([1.0, 1.0]))
check_grad(a.grad, [4.0, 5.0])  # b的值
check_grad(b.grad, [2.0, 3.0])  # a的值
a.zero_grad()
b.zero_grad()

# 测试除法
c = a / b
c.backward(np.array([1.0, 1.0]))
check_grad(a.grad, [1/4, 1/5])          # 1/b
check_grad(b.grad, [-2/(4**2), -3/(5**2)])  # -a/b²
a.zero_grad()
b.zero_grad()

print("===== 2. 广播测试 =====")
# 广播加法
a = Tensor([[1.0, 2.0]], requires_grad=True)  # 形状(1,2)
b = Tensor([[3.0], [4.0]], requires_grad=True)  # 形状(2,1)
c = a + b  # 形状(2,2)
c.backward(np.ones((2, 2)))
check_grad(a.grad, [[2.0, 2.0]])  # 沿0轴求和
check_grad(b.grad, [[2.0], [2.0]])  # 沿1轴求和
a.zero_grad()
b.zero_grad()

# 广播乘法
c = a * b
c.backward(np.ones((2, 2)))
check_grad(a.grad, [[3+4, 3+4]])  # b的和
check_grad(b.grad, [[1+2], [1+2]])  # a的和
a.zero_grad()
b.zero_grad()

print("===== 3. 矩阵乘法测试 =====")
a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # (2,2)
b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)  # (2,2)
c = a @ b
c.backward(np.ones((2, 2)))
# 验证a的梯度：ones @ b.T
expected_a_grad = np.ones((2,2)) @ b.data.T
check_grad(a.grad, expected_a_grad)
# 验证b的梯度：a.T @ ones
expected_b_grad = a.data.T @ np.ones((2,2))
check_grad(b.grad, expected_b_grad)
a.zero_grad()
b.zero_grad()

print("===== 4. 激活函数测试（exp、log） =====")
# 测试exp
a = Tensor([1.0, 2.0], requires_grad=True)
c = a.exp()
c.backward(np.array([1.0, 1.0]))
check_grad(a.grad, np.exp([1.0, 2.0]))  # exp(x)的导数是自身
a.zero_grad()

# 测试log
a = Tensor([2.0, 3.0], requires_grad=True)
c = a.log()
c.backward(np.array([1.0, 1.0]))
check_grad(a.grad, [1/2, 1/3])  # log(x)的导数是1/x
a.zero_grad()

print("===== 5. 聚合 聚合函数测试（sum、mean） =====")
# 测试sum
a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
c = a.sum(axis=0)  # 沿0轴求和
c.backward(np.array([1.0, 1.0]))
check_grad(a.grad, np.ones((2, 2)))  # 广播后全为1
a.zero_grad()

# 测试mean
c = a.mean(axis=1, keepdims=True)  # 沿1轴求平均
c.backward(np.ones((2, 1)))
expected_grad = np.ones((2, 2)) * (1/2)  # 平均梯度
check_grad(a.grad, expected_grad)
a.zero_grad()

print("Testing no_grad context manager...")
# 正常情况
x1 = Tensor([1, 2, 3], requires_grad=True)
assert x1.requires_grad == True, "正常情况 requires_grad 应该为 True"

# 在 no_grad 上下文中
with Tensor.no_grad():
    x2 = Tensor([1, 2, 3], requires_grad=True)
    assert x2.requires_grad == False, "no_grad 中 requires_grad 应该为 False"
    
    # 检查全局状态
    assert Tensor.is_grad_enabled() == False, "no_grad 中全局状态应该为 False"

# 离开 no_grad 后恢复
assert Tensor.is_grad_enabled() == True, "离开 no_grad 后全局状态应该恢复"

x3 = Tensor([1, 2, 3], requires_grad=True)
assert x3.requires_grad == True, "离开 no_grad 后 requires_grad 应该恢复"



# 创建测试张量
x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)

# 测试均值
mean_all = x.mean()
assert np.allclose(mean_all.data, 3.5), "全局均值计算错误"

mean_axis0 = x.mean(axis=0)
assert np.allclose(mean_axis0.data, [2.5, 3.5, 4.5]), "沿轴0均值计算错误"

mean_axis1 = x.mean(axis=1)
assert np.allclose(mean_axis1.data, [2, 5]), "沿轴1均值计算错误"

# 测试方差
var_all = x.var()
expected_var = np.var([[1, 2, 3], [4, 5, 6]], ddof=1)  # 样本方差
assert np.allclose(var_all.data, expected_var), "全局方差计算错误"

var_axis0 = x.var(axis=0)
expected_var_axis0 = np.var([[1, 2, 3], [4, 5, 6]], axis=0, ddof=1)
assert np.allclose(var_axis0.data, expected_var_axis0), "沿轴0方差计算错误"

# 测试标准差
std_all = x.std()
expected_std = np.std([[1, 2, 3], [4, 5, 6]], ddof=1)
assert np.allclose(std_all.data, expected_std), "全局标准差计算错误"


print("===== 6. 形状操作测试（reshape、transpose、expand_dims） =====")
# 测试reshape
a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
c = a.reshape(1, 4)
c.backward(np.array([[1.0, 1.0, 1.0, 1.0]]))
check_grad(a.grad, np.ones((2, 2)))  # 梯度形状还原
a.zero_grad()

# 测试transpose
c = a.transpose(1, 0)  # 转置
c.backward(np.ones((2, 2)))
check_grad(a.grad, np.ones((2, 2)))  # 梯度也转置回来
a.zero_grad()

# 测试expand_dims
c = a.expand_dims(axis=0)  # 新增0轴
c.backward(np.ones((1, 2, 2)))
check_grad(a.grad, np.ones((2, 2)))  # 梯度挤压新增维度
a.zero_grad()

print("===== 7. 切片与pad测试 =====")
# 测试切片
a = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
c = a[1:3]  # 取索引1和2
c.backward(np.array([1.0, 1.0]))
check_grad(a.grad, [0.0, 1.0, 1.0, 0.0])  # 切片位置梯度为1
a.zero_grad()

# 测试pad
a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
c = a.pad(((1, 1), (1, 1)))  # 四周各pad 1圈
c.backward(np.ones((4, 4)))
check_grad(a.grad, np.ones((2, 2)))  # 中间区域梯度为1
a.zero_grad()

print("=== 测试4D张量填充 ===")

# 创建4D张量 (batch, channels, height, width)
x = Tensor(np.ones((2, 3, 4, 4)), requires_grad=True)
print(f"原始形状: {x.shape}")

# 4D填充: ((batch前, batch后), (通道前, 通道后), (高度前, 高度后), (宽度前, 宽度后))
pad_width = ((0, 0), (0, 0), (1, 1), (1, 1))  # 在高度和宽度上各填充1

# 应用填充
x_padded = x.pad(pad_width)
print(f"填充后形状: {x_padded.shape}")

# 验证形状
expected_shape = (2, 3, 6, 6)  # 4+1+1=6, 4+1+1=6
assert x_padded.shape == expected_shape, f"期望 {expected_shape}, 实际 {x_padded.shape}"
print("✓ 形状正确")

# 验证填充值
# 中间区域应该是原始数据，边界应该是0
center_region = x_padded.data[:, :, 1:5, 1:5]  # 去除边界
assert np.allclose(center_region, 1.0), "中心区域值不正确"
print("✓ 填充值正确")

# 测试梯度传播
loss = x_padded.sum()
loss.backward()

# 检查梯度形状
assert x.grad.shape == x.shape, "梯度形状不正确"
print("✓ 梯度形状正确")

# 检查梯度值 - 应该只有中心区域有梯度
expected_grad = np.ones((2, 3, 4, 4))
assert np.allclose(x.grad, expected_grad), "梯度值不正确"
print("✓ 梯度值正确")

print("🎉 4D填充测试全部通过！")

print("===== 8. 重复操作测试（repeat） =====")
a = Tensor([[1.0, 2.0]], requires_grad=True)
c = a.repeat(repeats=2, axis=0)  # 沿0轴重复2次
c.backward(np.array([[1.0, 1.0], [1.0, 1.0]]))
check_grad(a.grad, [[2.0, 2.0]])  # 重复区域梯度求和
a.zero_grad()

print("===== 9. 极值测试（max、min） =====")
# 测试max
a = Tensor([[3.0, 1.0], [2.0, 4.0]], requires_grad=True)
c = a.max(axis=1)  # 沿1轴取最大值
c.backward(np.array([1.0, 1.0]))
expected_max_grad = np.zeros((2, 2))
expected_max_grad[0, 0] = 1.0  # 第0行最大值位置
expected_max_grad[1, 1] = 1.0  # 第1行最大值位置
check_grad(a.grad, expected_max_grad)
a.zero_grad()

# 测试min
c = a.min(axis=1)  # 沿1轴取最小值
c.backward(np.array([1.0, 1.0]))
expected_min_grad = np.zeros((2, 2))
expected_min_grad[0, 1] = 1.0  # 第0行最小值位置
expected_min_grad[1, 0] = 1.0  # 第1行最小值位置
check_grad(a.grad, expected_min_grad)
a.zero_grad()

print("===== 10. 链式传播测试 =====")
# 复杂计算链：z = (x*y + exp(x)) / mean(y)
x = Tensor([2.0, 3.0], requires_grad=True)
y = Tensor([4.0, 5.0], requires_grad=True)
z = (x * y + x.exp()) / y.mean()
z.backward(np.array([1.0, 1.0]))


n = y.size
mean_y = y.data.mean()
c_sum = (x.data * y.data + np.exp(x.data)).sum()
dx_expected = (y.data + np.exp(x.data)) / mean_y
dy_expected = (x.data / mean_y) - c_sum / (mean_y**2 * n)

check_grad(x.grad, dx_expected)
check_grad(y.grad, dy_expected)

print("所有测试通过！")