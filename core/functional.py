
from .tensor import Tensor

# 下面是运行funcional的时候取消注释
# from tensor import Tensor
import numpy as np

# ===== 一元激活 =====
def relu(x: Tensor, inplace=False) -> Tensor:
    """
    前向：out = max(0, x)
    反向：∂L/∂x = ∂L/∂out ⊙ (x>0)
    TODO：
        1. 计算 out_data
        2. 新建 Tensor out，挂起计算图
        3. 实现 _backward 回调，完成梯度回传
    """
    out_data = np.maximum(0, x.data)
    if inplace:
        return out_data
    out = Tensor(out_data, requires_grad=True)
    def _backward():
        if x.grad is not None:
            x.grad += out.grad * (x.data>0)
    out._backward = _backward
    out._parents = [x]
    return out

def sigmoid(x: Tensor, inplace=False) -> Tensor:
    """
    Sigmoid(x)=1/(1+e^(-x))
    反向：dL/dx = sigmoid(x)*(1-sigmoid(x))
    """
    # L 最终损失标量
    # ∂L/∂out = out.grad
    # ∂L/∂x = x.grad 
    # ∂L/∂y = other.grad
    # ∂out/∂x 使用numpy广播实现
    # out = sigmoid(x)
    # ∂L/∂x = ∂L/∂out *∂out/∂x = out.grad * sigmoid(x)*(1-sigmoid(x))
    
    out_data = 1/(1+np.exp(-x.data))
    if inplace:
        return out_data
    out = Tensor(out_data, requires_grad=True)
    def _backward():
        if x.grad is not None:
            x.grad += out.grad * (out_data*(1-out_data))
    out._backward = _backward
    out._parents = [x]
    return out
    
def tanh(x: Tensor, inplace=False) -> Tensor: 
    """
    tanh(x) = (e^(x)-e^(-x))/(e^(x)+e^(-x))
    """
    out_data = (np.exp(x.data)-np.exp(-x.data))/(np.exp(x.data)+np.exp(-x.data))
    if inplace:
        return out_data
    out = Tensor(out_data, requires_grad=True)
    # out = tanh(x)
    # ∂L/∂x = ∂L/∂out *∂out/∂x = out.grad * (1-tanh^2(x))
    def _backward():
        if x.grad is not None:
            x.grad += out.grad * (1-out_data*out_data)
    out._backward = _backward 
    out._parents = [x]
    return out



# ===== 归一化 =====
def softmax(x: Tensor, axis=-1) -> Tensor:
    """
    前向：数值稳定版 softmax
          max_val = x.max(axis, keepdims)
          x_stable = x - max_val
          exp_x = exp(x_stable)
          out = exp_x / exp_x.sum(axis, keepdims)

    反向：∂L/∂x = out * (grad_out - Σ(grad_out * out))
    TODO：
        1. 完成前向计算（复用 Tensor 的 max/exp/sum）
        2. 新建 Tensor out，挂计算图
        3. _backward 里按公式回传
    """

    """
    softmax(x_i) = exp(x_i) / sum(exp(x_j))
    反向：dL/dx = softmax * (dL/dout - sum(dL/dout * softmax))
    """
    # 1. 数值稳定：减最大值
    max_val = x.max(axis=axis, keepdims=True)          # 你得实现 x.max
    x_stable = x - max_val                             # broadcast 你已支持

    # 2. 指数
    exp_x = x_stable.exp()                             # 你得实现 Tensor.exp()

    # 3. 归一化
    sum_exp = exp_x.sum(axis=axis, keepdims=True)      # 你得实现 x.sum
    out = exp_x / sum_exp                              # broadcast 除法

    # out = softmax(x)
    # ∂L/∂x = ∂L/∂out *∂out/∂x
    # ∂L/∂x[i] = ∂L/∂out *∂out/∂x[i] =  Σ_j ∂L/∂out[j] · s[j](δ_{ij} − s[i])
    # = s[i] · (∂L/∂out[i] − Σ_j ∂L/∂out[j] · s[j])

    # 4. 链式回调：反向公式
    def _backward():
        if x.grad is not None:
            # 留空：根据链式规则完成梯度
            # 提示：grad_out = out.grad
            #       grad_x = out * (grad_out - (grad_out * out).sum(axis, keepdims=True))
            grad_out = out.grad
            s = (grad_out * out.data).sum(axis = axis, keepdims = True)
            x.grad += out.data * (grad_out - s)

    out._backward = _backward
    out._parents = [x]
    return out

def log_softmax(x: Tensor, axis=-1) -> Tensor:
    """
    前向：log(softmax(x)) = x - log(sum(exp(x)))
          仍需减 max 保证数值稳定
    反向：∂L/∂x = grad_out - exp(out)*grad_out.sum(axis, keepdims)
    TODO：
        1. 完成前向计算（复用 Tensor 的 max/exp/sum）
        2. 新建 Tensor out，挂计算图
        3. _backward 里按公式回传
    """

    # log_softmax(xᵢ) = ln(exp(xᵢ) / Σⱼ exp(xⱼ)) = xᵢ − ln(Σⱼ exp(xⱼ))
    # 数值稳定，减去最大值
    max_val = x.max(axis=axis, keepdims = True)
    x_stable = x - max_val
    # 指数+求和
    exp_x = x_stable.exp()
    sum_exp = exp_x.sum(axis = axis, keepdims = True)
    # ln(softmax) = x − ln(sum_exp)
    log_sum_exp = sum_exp.log() #补充tensor的log函数
    out = x_stable - log_sum_exp #需要使用广播减法
    # ∂L/∂x = ∂L/∂y ⊙ (1 − exp(y))  (y = log_softmax(x))
    def _backward():
        if x.requires_grad:
            if x.grad is None:
                x.grad = np.zeros_like(x.data)

            grad_out = out.grad #(N,C)
            exp_out = np.exp(out.data) #(N,C)
            # 对类别轴求和，保持维度
            # ∂L/∂x = ∂L/∂y - softmax(x) * sum(∂L/∂y)
            sum_grad = grad_out.sum(axis=axis, keepdims=True)  #(N,1)
            x_grad = grad_out - exp_out * sum_grad # 使用广播乘法

            if x.grad is None:
                x.grad = np.zeros_like(x.data)
            x.grad += x_grad
    out._backward = _backward
    out._parents = [x]

    return out


# ===== 损失 =====

def nll_loss(log_probs:Tensor, targets:Tensor)->'Tensor':

    """
    前向：
        N = log_probs.shape[0]
        idx  = targets.data.astype(int)   # 这里允许用 numpy 取值
        selected = log_probs[range(N), idx]   # 用 Tensor 索引保持图
        loss = -selected.mean()               # 返回标量 Tensor
    反向：
        无需手写，selected.mean() 会自动完成
    TODO：仅完成前向即可，计算图保持完整
    """

    """
    log_probs: (N, C)
    targets: (N,) int class indices
    """
    N = log_probs.shape[0]
    idx = targets.data.astype(int)
    selected = log_probs[range(N), idx]
    loss = -selected.mean()
    return loss

def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor: 
    """
    推荐实现：
        log_p = log_softmax(logits, axis=-1)
        return nll_loss(log_p, targets)
    这样无需手写 _backward；若坚持手动，可保留原硬编码版本。
    TODO：二选一
    """

    """
    logits:(N, C) raw score
    targets:(N,) class index
    """
    # log_p = log_softmax(logits, axis=-1)            # (N,C)
    # # 选取目标 log-prob
    # idx = targets.data.astype(int)
    # selected = log_p.data[np.arange(len(idx)),idx]  # (N,)
    # loss_data = -selected.mean()                    # scalar
    # out = Tensor(loss_data, requires_grad=True)
    # def _backward():
    #     # ∂L/∂logits = (softmax - one_hot) / N
    #     if logits is not None:
    #         p = logits.data.exp()/logits.data.exp().sum(axis = -1, keepdims = True)
    #         p[np.arange(len(idx)), idx] -= 1
    #         logits.grad += out.grad*p/len(idx)     #平均梯度

    # out._backward = _backward
    # out._parents = [logits, targets]
    # return out
    log_p = log_softmax(logits, axis= -1)

    # 添加数值检查
    # print(f"DEBUG cross_entropy: logits范围=[{logits.data.min():.4f}, {logits.data.max():.4f}]")
    # print(f"DEBUG: log_p范围=[{log_p.data.min():.4f}, {log_p.data.max():.4f}]")
    
    return nll_loss(log_p, targets=targets)

def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """
    前向：out = ((pred - target)**2).mean()
    反向：框架自动完成
    TODO：一行即可
    """
    # out_data = ((pred - target)**2).mean()
    # out = Tensor(out_data, requires_grad=True) #导致计算图中断

    # return ((pred - target)*(pred - target)).mean()
    return ((pred - target)**2).mean()

# ===== 卷积池化（CNN 阶段再写）=====
def conv2d(x: Tensor, w: Tensor, b: Tensor=None, stride=1, pad=0) -> Tensor: ...
def max_pool2d(x: Tensor, kernel_size=2, stride=2) -> Tensor: ...

# ===== 高级（LSTM/Transformer 阶段）=====
def lstm_cell(x: Tensor, hx: Tensor, cx: Tensor, w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor) -> (Tensor, Tensor): ...
def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor=None) -> Tensor: ...


if __name__=="__main__":

# 假设你的Tensor类已经定义在上面
# 这里只展示测试代码

    def test_activation_functions():
        print("===== 激活函数测试 =====")
        
        # 测试ReLU
        print("1. ReLU测试")
        x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
        y = relu(x)
        y.backward(np.ones_like(y.data))
        
        print(f"输入: {x.data}")
        print(f"ReLU输出: {y.data}")
        print(f"梯度: {x.grad}")
        expected_grad = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        assert np.allclose(x.grad, expected_grad), f"ReLU梯度错误: {x.grad} != {expected_grad}"
        print("ReLU测试通过 ✓")
        
        # 测试Sigmoid
        print("\n2. Sigmoid测试")
        x = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
        y = sigmoid(x)
        y.backward(np.ones_like(y.data))
        
        print(f"输入: {x.data}")
        print(f"Sigmoid输出: {y.data}")
        print(f"梯度: {x.grad}")
        
        # 手动计算期望梯度
        sigmoid_output = 1 / (1 + np.exp(-x.data))
        expected_grad = sigmoid_output * (1 - sigmoid_output)
        assert np.allclose(x.grad, expected_grad, atol=1e-6), f"Sigmoid梯度错误: {x.grad} != {expected_grad}"
        print("Sigmoid测试通过 ✓")
        
        # 测试Tanh
        print("\n3. Tanh测试")
        x = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
        y = tanh(x)
        y.backward(np.ones_like(y.data))
        
        print(f"输入: {x.data}")
        print(f"Tanh输出: {y.data}")
        print(f"梯度: {x.grad}")
        
        # 手动计算期望梯度
        tanh_output = np.tanh(x.data)
        expected_grad = 1 - tanh_output ** 2
        assert np.allclose(x.grad, expected_grad, atol=1e-6), f"Tanh梯度错误: {x.grad} != {expected_grad}"
        print("Tanh测试通过 ✓")

    def test_softmax():
        print("\n===== Softmax测试 =====")
        
        # 测试1: 基础softmax
        x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        y = softmax(x, axis=-1)
        y.backward(np.ones_like(y.data))
        
        print(f"输入: {x.data}")
        print(f"Softmax输出: {y.data}")
        print(f"输出和: {y.data.sum()}")
        print(f"梯度: {x.grad}")
        
        # 检查输出和为1
        assert abs(y.data.sum() - 1.0) < 1e-6, "Softmax输出和不为1"
        print("Softmax基础测试通过 ✓")
        
        # 测试2: 数值稳定性
        x = Tensor([[1000.0, 1000.0, 1000.0]], requires_grad=True)
        y = softmax(x, axis=-1)
        
        print(f"大数值输入: {x.data}")
        print(f"Softmax输出: {y.data}")
        assert not np.any(np.isnan(y.data)), "Softmax数值不稳定"
        print("Softmax数值稳定性测试通过 ✓")

    def test_log_softmax():
        print("\n===== Log Softmax测试 =====")
        
        x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        y = log_softmax(x, axis=-1)
        y.backward(np.ones_like(y.data))
        
        print(f"输入: {x.data}")
        print(f"Log Softmax输出: {y.data}")
        print(f"梯度: {x.grad}")
        
        # 验证 log_softmax(x) = log(softmax(x))
        softmax_out = softmax(x, axis=-1)
        expected = np.log(softmax_out.data + 1e-12)
        assert np.allclose(y.data, expected, atol=1e-6), "Log Softmax输出错误"
        print("Log Softmax测试通过 ✓")

    def test_loss_functions():
        print("\n===== 损失函数测试 =====")
        
        # 测试NLL Loss
        print("1. NLL Loss测试")
        log_probs = Tensor([
            [-1.0, -2.0, -3.0],  # 真实类在位置0
            [-3.0, -1.0, -2.0]   # 真实类在位置1
        ], requires_grad=True)
        targets = Tensor([0, 1])  # 类别索引
        
        loss = nll_loss(log_probs, targets)
        loss.backward()
        
        print(f"Log概率: {log_probs.data}")
        print(f"目标: {targets.data}")
        print(f"NLL Loss: {loss.data}")
        print(f"Log概率梯度: {log_probs.grad}")
        
        # 手动验证
        selected = np.array([log_probs.data[0, 0], log_probs.data[1, 1]])
        expected_loss = -selected.mean()
        assert abs(loss.data - expected_loss) < 1e-6, "NLL Loss计算错误"
        print("NLL Loss测试通过 ✓")
        
        # 测试Cross Entropy
        print("\n2. Cross Entropy测试")
        logits = Tensor([
            [2.0, 1.0, 0.1],  # 真实类在位置0
            [0.1, 2.0, 0.1]   # 真实类在位置1
        ], requires_grad=True)
        targets = Tensor([0, 1])
        
        loss = cross_entropy(logits, targets)
        loss.backward()
        
        print(f"Logits: {logits.data}")
        print(f"目标: {targets.data}")
        print(f"Cross Entropy Loss: {loss.data}")
        print(f"Logits梯度: {logits.grad}")
        
        # 验证梯度形状
        assert logits.grad.shape == logits.data.shape, "Cross Entropy梯度形状错误"
        print("Cross Entropy测试通过 ✓")
        
        # 测试MSE Loss
        print("\n3. MSE Loss测试")
        pred = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        target = Tensor([1.5, 1.8, 2.9])
        
        loss = mse_loss(pred, target)
        loss.backward()
        
        print(f"预测: {pred.data}")
        print(f"目标: {target.data}")
        print(f"MSE Loss: {loss.data}")
        print(f"预测梯度: {pred.grad}")
        
        # 手动验证
        expected_loss = ((pred.data - target.data) ** 2).mean()
        assert abs(loss.data - expected_loss) < 1e-6, "MSE Loss计算错误"
        print("MSE Loss测试通过 ✓")

    def test_complex_chain():
        print("\n===== 复杂链式测试 =====")
        
        # 构建一个简单的神经网络前向传播
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        w = Tensor([[0.5, -0.5], [0.1, 0.9]], requires_grad=True)
        b = Tensor([0.1, 0.2], requires_grad=True)
        
        # 前向传播
        linear = x @ w + b  # 矩阵乘法 + 偏置
        activated = relu(linear)  # ReLU激活
        normalized = softmax(activated, axis=-1)  # Softmax归一化
        
        # 计算损失
        targets = Tensor([0, 1])  # 两个样本的真实类别
        loss = cross_entropy(normalized, targets)
        
        print(f"输入形状: {x.shape}")
        print(f"线性输出形状: {linear.shape}")
        print(f"激活输出形状: {activated.shape}")
        print(f"归一化输出形状: {normalized.shape}")
        print(f"损失: {loss.data}")
        
        # 反向传播
        loss.backward()
        
        # 检查梯度是否存在
        assert x.grad is not None, "输入梯度未计算"
        assert w.grad is not None, "权重梯度未计算"
        assert b.grad is not None, "偏置梯度未计算"
        
        print(f"输入梯度形状: {x.grad.shape}")
        print(f"权重梯度形状: {w.grad.shape}")
        print(f"偏置梯度形状: {b.grad.shape}")
        
        print("复杂链式测试通过 ✓")

    
    # 运行所有测试
    test_activation_functions()
    test_softmax()
    test_log_softmax()
    test_loss_functions()
    test_complex_chain()
    
    print("\n🎉 所有测试通过！")