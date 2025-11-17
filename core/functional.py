
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
def conv2d(x: Tensor, weight: Tensor, bias: Tensor = None, stride=1, padding=0) -> Tensor:
    """
    2D卷积的函数式实现
    
    参数:
        x: 输入张量 (batch_size, in_channels, height, width)
        weight: 卷积核 (out_channels, in_channels, kernel_height, kernel_width)
        bias: 偏置 (out_channels,)
        stride: 步长
        padding: 填充
    
    返回:
        输出张量 (batch_size, out_channels, out_height, out_width)
    """
    # 1.处理参数
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding

    batch_size, in_channels, in_h, in_w = x.shape
    out_channels, _, kH, kW = weight.shape
   
    # 2.计算输出尺寸
    out_h = (in_h + 2 * padding[0] - kH) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - kW) // stride[1] + 1
    
    # 3.应用填充
    if padding[0]>0 or padding[1]>0:
        x_padded = x.pad(((0,0),(0,0),(padding[0],padding[0]),(padding[1],padding[1])))
    else:
        x_padded = x
    # 4.im2col
    # 形状：(in_channels * kH * kW, batch_size * out_h * out_w)
    cols = _im2col(x_padded.data, kH, kW, stride, out_h, out_w)
    # 5.重塑权重进行矩阵乘法
    weight_flat = weight.data.reshape(out_channels, -1) # (out_channels, in_channels * kH * kW)
    # 6.执行卷积
    out_data = weight_flat@cols# (out_channels, batch_size * out_h * out_w)
    # 7.重塑输出
    out_data = out_data.reshape(out_channels, batch_size, out_h,out_w)
    out_data = out_data.transpose(1,0,2,3)# (batch_size, out_channels, out_h, out_w)
    
    # 8.添加偏置
    if bias is not None:
        out_data += bias.data.reshape(1,-1,1,1)
    out = Tensor(out_data, requires_grad=x.requires_grad or weight.requires_grad or (bias and bias.requires_grad))

    # 9.保存中间结果进行反向传播
    def _backward():
        if x.requires_grad or weight.requires_grad or (bias and bias.requires_grad):
            grad_out = out.grad # (batch_size, out_channels, out_h, out_w)
            # 重塑梯度以便得到矩阵
            grad_out_reshape = grad_out.transpose(1,0,2,3) # (out_channels, batch_size, out_h, out_w)
            grad_out_flat = grad_out_reshape.reshape(out_channels, -1) # (out_channels, batch_size * out_h * out_w)
            # 计算梯度权重
            if weight.requires_grad:
                # dL/dW = dL/dout * cols^T
                # print(f"cols: min={cols.min()}, max={cols.max()}, mean={cols.mean()}")
                grad_weight_flat = grad_out_flat@cols.T # (out_channels, in_channels * kH * kW)
                # print(f"grad_weight_flat: min={grad_weight_flat.min()}, max={grad_weight_flat.max()}, mean={grad_weight_flat.mean()}")
                grad_weight = grad_weight_flat.reshape(weight.shape)
                if weight.grad is None:
                    weight.grad = np.zeros_like(weight.data)
                weight.grad += grad_weight
                # print(f"weight.grad: min={weight.grad.min()}, max={weight.grad.max()}, mean={weight.grad.mean()}")
            # 计算梯度偏置
            if bias and bias.requires_grad:
                # dL/db = sum(dL/dout, axis=(0, 2, 3))
                grad_bias = grad_out_flat.sum(axis=1)
                if bias.grad is None:
                    bias.grad = np.zeros_like(bias.data)
                bias.grad += grad_bias
                # print(f"bias.grad: min={bias.grad.min()}, max={bias.grad.max()}, mean={bias.grad.mean()}")

            # 计算输入梯度
            if x.requires_grad:
                # dL/dx = W^T * dL/dout 然后 col2im
                weight_flat = weight.data.reshape(out_channels, -1) # (out_channels, in_channels * kH * kW)
                grad_cols = weight_flat.T @ grad_out_flat # (in_channels * kH * kW, batch_size * out_h * out_w)

                # 将列转化为图像格式
                grad_x_padded = _col2im(grad_cols, x_padded.shape, kH ,kW, stride, out_h, out_w)

                # 除去padding
                if padding[0]>0 or padding[1]>0:
                    grad_x = grad_x_padded[:,:,padding[0]:padding[0]+in_h,padding[1]:padding[1]+in_w]
                else:
                    grad_x = grad_x_padded
                if x.grad is None:
                    x.grad = np.zeros_like(x.data)
                x.grad += grad_x
    out._backward = _backward
    out._parents = [x,weight]
    if bias is not None:
        out._parents.append(bias)
    return out


def _im2col(x, kH, kW, stride, out_h, out_w):
    """
    将输入图像转换为列矩阵
    x: (batch_size, in_channels, padded_h, padded_w)
    返回: (in_channels * kH * kW, batch_size * out_h * out_w)
    """
    batch_size, in_channels, padded_h, padded_w = x.shape
    sH, sW = stride

    cols = np.zeros((in_channels*kH*kW,batch_size*out_h*out_w))
    col_idx = 0
    for b in range(batch_size):
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * sH
                w_start = j * sW
                patch = x[b,:,h_start:h_start+kH,w_start:w_start+kW]
                cols[:,col_idx] = patch.reshape(-1)
                col_idx +=1
    return cols

def _col2im(cols, x_shape, kH, kW, stride, out_h, out_w):
    """
    将列矩阵转换回图像格式
    """
    batch_size, in_channels, padded_h, padded_w = x_shape
    sH, sW = stride
    # dx = np.zeros((batch_size, in_channels, padded_h, padded_w))
    dx = np.zeros(x_shape, dtype=cols.dtype)
    col_idx = 0
    for b in range(batch_size):
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * sH
                w_start = j * sW
                path_grad = cols[:,col_idx].reshape(in_channels, kH, kW)
                dx[b,:,h_start:h_start+kH,w_start:w_start+kW]+= path_grad
                col_idx +=1
    return dx


def max_pool2d(x: Tensor, kernel_size=2, stride=None, padding=0) -> Tensor:
    """
    2D最大池化的函数式实现
    
    参数:
        x: 输入张量 (batch_size, channels, height, width)
        kernel_size: 池化核大小
        stride: 步长（默认等于kernel_size）
        padding: 填充
    
    返回:
        输出张量 (batch_size, channels, out_height, out_width)
    """
    # 1.参数处理
    kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    stride = stride if stride is not None else kernel_size
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding
    batch_size, channels, in_h, in_w = x.shape
    kH, kW = kernel_size
    sH, sW = stride

    # 计算输出的尺寸
    out_h = (in_h + 2*padding[0] - kH)//sH + 1
    out_w = (in_w + 2*padding[1] - kW)//sW + 1

    # 应用填充
    if padding[0]>0 or padding[1]>0:
        x_padded = x.pad(((0,0),(0,0),(padding[0],padding[0]),(padding[1],padding[1])))
    else:
        x_padded= x
    
    # 初始化输出和最大索引
    out_data = np.zeros((batch_size,channels,out_h,out_w))
    max_indices = np.zeros((batch_size, channels,out_h,out_w,2),dtype=int)

    # 执行最大化池
    for b in range(batch_size):
        for c in range(channels):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i *sH #需要池化的左上角
                    w_start= j * sW

                    patch = x_padded.data[b,c,h_start:h_start+kH,w_start:w_start+kW]

                    # 找到最大值以及索引
                    max_val= np.max(patch)
                    max_idx = np.unravel_index(np.argmax(patch), patch.shape)

                    out_data[b,c,i,j] = max_val
                    max_indices[b,c,i,j] = [h_start+max_idx[0], w_start+max_idx[1]]
    
    out = Tensor(out_data,requires_grad=x.requires_grad)
    # 保留中间结果用于方向传播
    def _backward():
        if x.requires_grad:
            grad_out = out.grad
            # batch_size, channels, out_h, out_w = grad_out.shape
            # 初始化输入梯度
            grad_x= np.zeros(x_padded.shape)
            # 将梯度传播到最大值的位置
            for b in range(batch_size):
                for c in range(channels):
                    for i in range(out_h):
                        for j in range(out_w):
                            h_idx, w_idx = max_indices[b,c,i,j] #获取前向时的最大值的位置
                            grad_x[b,c,h_idx,w_idx]+=grad_out[b,c,i,j] #梯度只传回最大值的位置
            # 去除padding
            if padding[0]>0 or padding[1]>0:
                grad_x = grad_x[:,:,padding[0]:padding[0]+in_h,padding[1]:padding[1]+in_w]
            if x.grad is None:
                x.grad = np.zeros_like(x.data)
            x.grad+=grad_x
    out._backward = _backward
    out._parents = [x]
    return out

# ===== 高级（LSTM/Transformer 阶段）=====
def lstm_cell(x: Tensor, hx: Tensor, cx: Tensor, w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor) -> (Tensor, Tensor): ...

def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor=None) -> Tensor: ...

