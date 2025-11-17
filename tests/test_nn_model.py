
import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nn import Linear, Dropout, BatchNorm1d, MaxPool2d, Conv2d, Flatten, BatchNorm2d
from core import Tensor
from nn import Module, Sequential
print("Model 测试")
print("=" * 50)


# 创建一个测试模块
class TestModule(Module):
    def __init__(self):
        super().__init__()
        # 这些应该被正确注册
        self.linear = Linear(10, 5)  # 假设 Linear 继承自 Module
        self.custom_param = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        self.normal_attr = "hello"
    
    def forward(self, x):
        return self.linear(x)

try:
    test_module = TestModule()
    print("✓ TestModule 实例化成功")
    
    # 测试属性访问
    print(f"✓ 访问 linear: {type(test_module.linear).__name__}")
    print(f"✓ 访问 custom_param: {test_module.custom_param.shape}")
    print(f"✓ 访问 normal_attr: {test_module.normal_attr}")
    print(f"✓ 访问 training: {test_module.training}")
    
    # 测试参数收集
    params = list(test_module.parameters())
    print(f"✓ 参数收集: 找到 {len(params)} 个参数")
    
    # 测试模块收集
    modules = list(test_module.modules())
    print(f"✓ 模块收集: 找到 {len(modules)} 个模块")
    
    # 测试子模块收集
    children = list(test_module.children())
    print(f"✓ 子模块收集: 找到 {len(children)} 个子模块")
    
    print("🎉 所有测试通过！")
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()


# 测试 1: Module 基类功能
print("\n1. Module 基类功能测试")

class TestModule(Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(10, 5)
        self.dropout = Dropout(0.5)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        return x

test_module = TestModule()
print(f"创建测试模块: {test_module}")

# 测试参数收集


params = list(test_module.parameters())
print(f"参数数量: {len(params)}")
for i, param in enumerate(params):
    print(f"参数 {i}: 形状={param.shape}")

assert len(params) == 2, "参数数量错误"
print("✓ Module 基类测试通过")

# 测试 2: Sequential 容器功能
print("\n2. Sequential 容器功能测试")

# 创建 Sequential 模型
model = Sequential(
    Linear(10, 20),
    Linear(20, 10),
    Linear(10, 5)
)

print(f"创建 Sequential 模型: {model}")
print(f"模型层数: {len(model)}")

# 测试前向传播
x = Tensor(np.random.randn(32, 10), requires_grad=True)
output = model(x)
print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
assert output.shape == (32, 5), "输出形状错误"
print("✓ Sequential 前向传播测试通过")

# 测试参数收集
model_params = list(model.parameters())
print(f"Sequential 参数数量: {len(model_params)}")
assert len(model_params) == 6, "Sequential 参数数量错误"  # 3个线性层 × 2个参数
print("✓ Sequential 参数收集测试通过")

# 测试索引访问
first_layer = model[0]
print(f"第一层: {first_layer}")
assert isinstance(first_layer, Linear), "索引访问错误"

# 测试切片
first_two_layers = model[0:2]
print(f"前两层: {first_two_layers}")
assert len(first_two_layers) == 2, "切片访问错误"
print("✓ Sequential 索引和切片测试通过")

# 测试追加层
model.append(Linear(5, 2))
print(f"追加层后模型: {model}")
print(f"追加层后层数: {len(model)}")
assert len(model) == 4, "追加层错误"

# 测试追加层后的前向传播
output_after_append = model(x)
print(f"追加层后输出形状: {output_after_append.shape}")
assert output_after_append.shape == (32, 2), "追加层后输出形状错误"
print("✓ Sequential 追加层测试通过")

# 测试 3: 训练/评估模式
print("\n3. 训练/评估模式测试")

# 检查初始模式
assert model.training == True, "初始模式不是训练模式"

# 切换到评估模式
model.eval()
assert model.training == False, "切换到评估模式失败"

# 检查所有子模块的模式
for i, layer in enumerate(model):
    if hasattr(layer, 'training'):
        assert layer.training == False, f"第 {i} 层评估模式设置失败"

# 切换回训练模式
model.train()
assert model.training == True, "切换回训练模式失败"

for i, layer in enumerate(model):
    if hasattr(layer, 'training'):
        assert layer.training == True, f"第 {i} 层训练模式设置失败"

print("✓ 训练/评估模式测试通过")

# 测试 4: 梯度清零
print("\n4. 梯度清零测试")

# 模拟梯度计算
for param in model.parameters():
    param.grad = np.ones_like(param.data)

# 检查梯度是否存在
has_gradients = any(param.grad is not None for param in model.parameters())
assert has_gradients, "梯度未设置"

# 清零梯度
model.zero_grad()

# 检查梯度是否被清零
all_gradients_zero = all(
    param.grad is None or np.allclose(param.grad, 0) 
    for param in model.parameters()
)
assert all_gradients_zero, "梯度未清零"
print("✓ 梯度清零测试通过")

# 测试 5: 状态字典
print("\n5. 状态字典测试")

state_dict = model.state_dict()
print(f"状态字典键: {list(state_dict.keys())}")

# 检查状态字典是否包含所有参数
expected_keys = ['0.weight', '0.bias', '1.weight', '1.bias', 
                    '2.weight', '2.bias', '3.weight', '3.bias']
for key in expected_keys:
    assert key in state_dict, f"状态字典缺少键: {key}"

print("✓ 状态字典测试通过")

# 测试 6: 命名参数
print("\n6. 命名参数测试")

named_params = list(model.named_parameters())
print(f"命名参数数量: {len(named_params)}")

for name, param in named_params:
    print(f"  {name}: {param.shape}")

# 检查命名是否正确
param_names = [name for name, param in named_params]
expected_names = [
    '0.weight', '0.bias',
    '1.weight', '1.bias', 
    '2.weight', '2.bias',
    '3.weight', '3.bias'
]

for expected_name in expected_names:
    assert expected_name in param_names, f"缺少命名参数: {expected_name}"

print("✓ 命名参数测试通过")

print("\n" + "=" * 50)
print("所有 Model 测试通过！🎉")