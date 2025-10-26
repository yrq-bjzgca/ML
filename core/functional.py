from .tensor import Tensor
import numpy as np

# ===== 一元激活 =====
def relu(x: Tensor) -> Tensor:
    out_data = np.maxium(0, x.data)
    out = Tensor(out_data, requires_grad=True)
    def _backward():
        if x.grad is not None:
            x.grad += out.grad * (x.data>0)
    out._backward = _backward
    out._parents = [x]
    return out

def sigmoid(x: Tensor) -> Tensor:
    """
    Sigmoid(x)=1/(1+e^(-x))
    反向：dL/dx = sigmoid(x)*(1-sigmoid(x))
    """
    out_data = 1/(1+np.exp(-x.data))
    out = Tensor(out_data, requires_grad=True)
    def _backward():
        if x.grad is not None:
            x.grad += out.grad * (out_data*(1-out_data))
    out._backward = _backward
    out._parents = [x]
    return out
    
def tanh(x: Tensor) -> Tensor: 
    """
    tanh(x) = (e^(x)-e^(-x))/(e^(x)+e^(-x))
    """
    out_data = (np.exp(x.data)-np.exp(-x.data))/(np.exp(x.data)+np.exp(-x.data))
    out = Tensor(out_data, requires_grad=True)
    def _backward():
        if x.grad is not None:
            x.grad += out.grad * (1-out_data*out_data)
    out._backward = _backward 
    out._parents = [x]
    return out



# ===== 归一化 =====
def softmax(x: Tensor, axis=-1) -> Tensor:
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

    # 4. 链式回调：反向公式
    def _backward():
        if x.grad is not None:
            # 留空：根据链式规则完成梯度
            # 提示：grad_out = out.grad
            #       grad_x = out * (grad_out - (grad_out * out).sum(axis, keepdims=True))
            grad_out = out.grad
            s = (grad_out * out.data).sum(axis = axis, keepdim = True)
            x.grad += out.data * (grad_out - s)

    out._backward = _backward
    out._parents = [x]
    return out

def log_softmax(x: Tensor, axis=-1) -> Tensor: ...

# ===== 损失 =====
def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor: ...
def mse_loss(pred: Tensor, target: Tensor) -> Tensor: ...

# ===== 卷积池化（CNN 阶段再写）=====
def conv2d(x: Tensor, w: Tensor, b: Tensor=None, stride=1, pad=0) -> Tensor: ...
def max_pool2d(x: Tensor, kernel_size=2, stride=2) -> Tensor: ...

# ===== 高级（LSTM/Transformer 阶段）=====
def lstm_cell(x: Tensor, hx: Tensor, cx: Tensor, w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor) -> (Tensor, Tensor): ...
def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor=None) -> Tensor: ...