import numpy as np
from typing import List, Optional, Union, Tuple
import contextlib
class Tensor:
    """
    手写可微张量
    要求：支持广播、切片、pad；记录计算图；链式反向
    """
    # 类变量，控制全局梯度计算
    _grad_enabled = True

    def __init__(self, data, requires_grad=False):
        self.data = np.asarray(data, dtype=np.float32)
        self.shape = self.data.shape
        self.requires_grad = requires_grad and Tensor._grad_enabled
        self.grad = None
        if requires_grad:
            self.grad = np.zeros_like(self.data)
        self._backward = lambda: None #反向函数
        self._parents = [] #计算图父亲
        
    # ---------- 工具 ----------
    def __repr__(self):
        return f"Tensor({self.data}, shape={self.shape}, requires_grad={self.grad is not None})"

    @classmethod
    @contextlib.contextmanager
    def no_grad(cls):
        """
        上下文管理器，禁用梯度计算
        
        用法:
            with Tensor.no_grad():
                # 在这个块中创建的张量不会计算梯度
                x = Tensor([1, 2, 3], requires_grad=True)  # 实际 requires_grad=False
        """
        old_state = cls._grad_enabled
        cls._grad_enabled = False
        try:
            yield
        finally:
            cls._grad_enabled = old_state
    
    @classmethod
    def set_grad_enabled(cls, mode: bool):
        """
        设置梯度计算是否启用
        
        参数:
            mode: True 启用梯度计算，False 禁用
        """
        cls._grad_enabled = mode
    
    @classmethod
    def is_grad_enabled(cls):
        """
        检查梯度计算是否启用
        
        返回:
            bool: 如果启用梯度计算返回 True，否则返回 False
        """
        return cls._grad_enabled
    
    # @property
    # def requires_grad(self):
    #     # 判断是不是需要计算梯度
    #     return self.grad is not None

    @property
    def ndim(self):
        """返回张量的维度数"""
        return self.data.ndim
    

    def copy(self) -> 'Tensor':
        """
        创建当前张量的副本
        
        返回:
            新的 Tensor 对象，包含数据的副本
        """

        # 复制数据
        data_copy = self.data.copy()

        # 创建新的tensor
        new_tensor = Tensor(data_copy, requires_grad=self.requires_grad)

        # 如果原始张量有梯度，也复制梯度
        if self.grad is not None:
            new_tensor.grad = self.grad.copy()
        return new_tensor
    
    # 优化后的轴计算（__add__ 和 __mul__ 中均适用）
    # def get_broadcast_axes(self, grad_shape, target_shape):
    #     """计算广播新增的轴（需求和的轴）"""
    #     grad_ndim = len(grad_shape)
    #     target_ndim = len(target_shape)
    #     max_ndim = max(grad_ndim, target_ndim)

    #     # 补1使两者维度数一致（便于逐位）
    #     grad_shape_padded = (1,) * (max_ndim - grad_ndim) + grad_shape
    #     target_shape_padded = (1,) * (max_ndim - target_ndim) + target_shape
        
    #     # if grad_ndim > target_ndim:
    #     #     return tuple(range(grad_ndim - target_ndim))
    #     # if grad_ndim < target_ndim:
    #     #     return tuple(range(grad_ndim, target_ndim))
    #     # if grad_ndim == target_ndim:
    #     #     return ()
    #     # 新增的轴是 grad 比 target 多的前 N 个轴
    #     # return tuple(range(grad_ndim - target_ndim))

    #     axes = []
    #     for i in range(max_ndim):
    #         g = grad_shape_padded[i]
    #         t = target_shape_padded[i]
    #         # 若梯度在该轴的尺寸 > 目标尺寸，且目标尺寸为1（说明是广播扩展的轴）
    #         if g > t and t == 1:
    #             axes.append(i)
    #     return tuple(axes)

    def get_broadcast_axes(self, grad_shape, target_shape):
        # 计算广播轴
        grad_ndim = len(grad_shape)
        target_ndim = len(target_shape)

        # 处理维度不一致的情况
        if grad_ndim > target_ndim:
            return tuple(range(grad_ndim - target_ndim))
        axes = []

        # 从右向左广播
        for i in range(1,min(grad_ndim, target_ndim)+1):
            grad_dim = grad_shape[-i]
            target_dim = target_shape[-i]
            if grad_dim != target_dim and target_dim ==1:
                axes.append(grad_ndim - i)
        return tuple(axes)

    def __add__(self, other:'Tensor')->'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other,requires_grad=False)
        # 带有广播
        a,b  = np.broadcast_arrays(self.data, other.data)
        out = Tensor(a + b, requires_grad=self.requires_grad or other.requires_grad)
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    # grad_broadcast = out.grad
                    self.grad = np.zeros_like(self.data)
                #计算广播轴求和
                axes = self.get_broadcast_axes(out.grad.shape, self.shape)
                if axes:
                    summed_grad = out.grad.sum(axis =axes, keepdims =True)
                    if summed_grad.shape!=self.shape:
                        # 尝试reshape到目标形状
                        try:
                            reshaped_grad = summed_grad.reshape(self.shape)
                            self.grad += reshaped_grad
                        except ValueError:
                            self.grad += np.broadcast_to(summed_grad, self.shape)
                    else:
                        self.grad += summed_grad
                else:
                    #没有需要相加的和
                    self.grad += out.grad
            
            if other.requires_grad:
                if other.grad is None:
                    # grad_broadcast = out.
                    other.grad = np.zeros_like(other.data)
                # 计算广播轴求和
                axes = self.get_broadcast_axes(out.grad.shape, other.shape)
                if axes:
                    summed_grad = out.grad.sum(axis =axes, keepdims =True)
                    if summed_grad.shape!=other.shape:
                        # 尝试reshape到目标形状
                        try:
                            reshaped_grad = summed_grad.reshape(other.shape)
                            other.grad += reshaped_grad
                        except ValueError:
                            other.grad += np.broadcast_to(summed_grad, other.shape)
                    else:
                        other.grad += summed_grad
                else:
                    #没有需要相加的和
                    other.grad += out.grad
        out._backward = _backward
        out._parents = [self, other]
        return out
    
    # 减法
    def __sub__(self, other:'Tensor')->'Tensor':
        other = other if isinstance(other,Tensor) else Tensor(other)
        return self + (-other)
    
    #对张量取负数
    def __neg__(self)->'Tensor':
        out = Tensor(-self.data, requires_grad=self.grad is not None)
        def _backward():
            # if self.grad is not None:
            if self.requires_grad and self.grad is not None:
                self.grad += -out.grad
        out._backward = _backward
        out._parents = [self]
        return out
    
    def __mul__(self, other:'Tensor')->'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other,requires_grad=False)
        # L 最终损失标量
        # ∂L/∂out = out.grad
        # ∂L/∂x = self.grad 
        # ∂L/∂y = other.grad
        # ∂out/∂x 使用numpy广播实现
        # out = x * y
        # ∂L/∂x = ∂L/∂out * y
        # ∂L/∂y = ∂L/∂out * x
        
        #原始版本
        # out = Tensor(self.data * other.data, requires_grad= True)
        # def _backward():
        #     if self.grad is not None:
        #         self.grad += out.grad * other.data
        #     if other.grad is not None:
        #         other.grad += out.grad * self.data

        # out._backward =  _backward
        # out._parents = [self,other]
        # return out
        # 带有广播
        a, b = np.broadcast_arrays(self.data, other.data)
        out = Tensor(a * b, requires_grad= self.requires_grad or other.requires_grad)
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                grad_broad = out.grad * b
                axes = self.get_broadcast_axes(grad_broad.shape, self.shape)
                if axes:
                    summed_grad = grad_broad.sum(axis =axes, keepdims =True)
                    if summed_grad.shape!=self.shape:
                        # 尝试reshape到目标形状
                        try:
                            reshaped_grad = summed_grad.reshape(self.shape)
                            self.grad += reshaped_grad
                        except ValueError:
                            self.grad += np.broadcast_to(summed_grad, self.shape)
                    else:
                        self.grad += summed_grad
                else:
                    #没有需要相加的和
                    self.grad += grad_broad
                # self.grad += grad_broad.sum(axis=axis).reshape(self.shape)
                # summed_grad = grad_broad.sum(axis =axes, keepdims =True)
                # self.grad += np.broadcast_to(summed_grad, self.shape)
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                grad_broad = out.grad * a
                axes = self.get_broadcast_axes(grad_broad.shape, other.shape)
                if axes:
                    summed_grad = grad_broad.sum(axis =axes, keepdims =True)
                    if summed_grad.shape!=other.shape:
                        # 尝试reshape到目标形状
                        try:
                            reshaped_grad = summed_grad.reshape(other.shape)
                            other.grad += reshaped_grad
                        except ValueError:
                            other.grad += np.broadcast_to(summed_grad, other.shape)
                    else:
                        other.grad += summed_grad
                else:
                        other.grad += grad_broad
                # other.grad += grad_broad.sum(axis=axis).reshape(other.shape)
                # summed_grad = grad_broad.sum(axis =axes, keepdims =True)
                # other.grad += np.broadcast_to(summed_grad, other.shape)                

        out._backward = _backward
        out._parents = [self, other]
        return out

    def __pow__(self, other:'Tensor')->'Tensor':
        """
        重载 ** 运算符
        目前支持：
            - self 为任意张量
            - other 为常数或 0-D/1-D 常量张量（广播）
        反向：
            ∂L/∂x = ∂L/∂out * other * x^(other-1)
            ∂L/∂y = ∂L/∂out * x^y * log(x)   （y 为张量时才需要）
        """
        other = other if isinstance(other, Tensor) else Tensor(other,requires_grad=False)
        out_data = np.pow(self.data,other.data)
        out = Tensor(out_data, requires_grad= (self.requires_grad or other.requires_grad))
        def _backward():
            if self.requires_grad:

                grad_self = out.grad * other.data * np.pow(self.data,other.data-1) 
                if grad_self.shape != self.shape:
                    axes = self.get_broadcast_axes(grad_self.shape, self.shape)
                    grad_self = grad_self.sum(axis=axes, keepdims=True)
                    grad_self = np.broadcast_to(grad_self, self.shape)
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)

                # summed_grad = grad_broad.sum(axis =axes, keepdims =True)
                # self.grad += np.broadcast_to(summed_grad, self.grad_broad)
                self.grad += grad_self

            if other.requires_grad:   
                grad_other = out.grad * other.data * np.log(self.data + 1e-12)# 防止log(0)
                if grad_other.shape != other.shape:
                    axes = self.get_broadcast_axes(grad_other.shape, other.shape)
                    grad_other = grad_self.sum(axis=axes, keepdims=True)
                    grad_other = np.broadcast_to(grad_other, other.shape)   
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                # grad_broad = out.grad * other.data * np.pow(self.data,other.data-1) 
                # axes = self.get_broadcast_axes(grad_broad.shape, other.shape)
                # summed_grad = grad_broad.sum(axis =axes, keepdims =True)
                # other.grad += np.broadcast_to(summed_grad, other.grad_broad)
                    other.grad += grad_other

        out._backward = _backward
        out._parents = [self, other]
        return out
    
    def sqrt(self):
        return self ** 0.5
    
    def square(self):
        """
        对self**2 的有好的接口
        """
        return self**2
    
    # 除法
    def __truediv__(self, other:'Tensor')->'Tensor':
        other  = other if isinstance(other,Tensor) else Tensor(other, requires_grad=False)
        a, b = np.broadcast_arrays(self.data, other.data)
        out = Tensor(a/b, requires_grad=self.requires_grad or other.requires_grad)
        def _backeard():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                # 自身梯度计算 (dL/dx = dL/dout * 1/b)
                grad_broad = out.grad/b
                axes = self.get_broadcast_axes(grad_broad.shape, self.shape)
                self.grad += grad_broad.sum(axis=axes).reshape(self.shape)
                # summed_grad = grad_broad.sum(axis=axes, keepdims=True)
                # self.grad += np.broadcast_to(summed_grad, self.shape)

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                # 其他张量梯度计算 (dL/dy = -dL/dout * x/b²)
                grad_broad = -out.grad*a/(b**2+1e-12)
                axes = self.get_broadcast_axes(grad_broad.shape, other.shape)
                other.grad += grad_broad.sum(axis=axes).reshape(other.shape)
                # summed_grad = grad_broad.sum(axis=axes, keepdims=True)
                # other.grad += np.broadcast_to(summed_grad, other.shape)
               

        out._backward = _backeard
        out._parents = [self, other]
        return out

    def exp(self:'Tensor')->'Tensor':
        # 带有广播的,不写broadcast_arrays是因为np中自带的广播的机制
        out_data = np.exp(self.data)
        out = Tensor(out_data, requires_grad= self.requires_grad)
        def _backward():
            if self.requires_grad and self.grad is not None:
                self.grad += out.grad * out.data
        out._backward = _backward
        out._parents = [self]
        return out
            
    def log(self:'Tensor')->'Tensor':
        # 带有广播的
        eps = 1e-12
        out_data = np.log(self.data+eps)
        out = Tensor(out_data, requires_grad= self.requires_grad)
        def _backward():
            if self.grad is not None:
                # 带有保护措施
                self.grad += out.grad / (self.data + eps)
        out._backward = _backward
        out._parents = [self]
        return out
    
    # 两个矩阵的乘法
    def __matmul__(self, other:'Tensor', axis=None)->'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other,requires_grad=False)
        batch_self,batch_other = self.data.shape[:-2],other.data.shape[:-2]
        # -2是高， -1是宽
        m, k1 = self.data.shape[-2], self.data.shape[-1]
        k2, n = other.data.shape[-2], other.data.shape[-1]

        if k1!=k2:
            raise ValueError(f"matul shape error, mismatch:{self.shape} and {other.shape}, by yrq")
        out_data = self.data@other.data#使用numpy进行操纵
        out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad)
        # A:(…, m, k)
        # B:(…, k, n)
        # C:A @ B 形状：(…, m, n)
        # G := ∂L/∂C 形状：(…, m, n)（上游梯度）
        # 对A的梯度：∂L/∂A = G @ Bᵀ
        # 对B的梯度：∂L/∂B = Aᵀ @ G
        def _backward():    
            if self.requires_grad:
                if self.grad is None:
                    # ∂L/∂A = G · Bᵀ
                    self.grad = np.zeros_like(self.data)
                grad_broad = out.grad @ other.data.swapaxes(-1,-2)

                # print(f"DEBUG matmul grad: self.shape={self.shape}, other.shape={other.shape}")
                # print(f"DEBUG: out.grad.shape={out.grad.shape}, grad_broad.shape={grad_broad.shape}")
                
                if grad_broad.shape != self.shape:
                    # axis = tuple(range(grad_broad.ndim - self.ndim))
                    axes = self.get_broadcast_axes(grad_broad.shape, self.shape)
                    grad_broad = grad_broad.sum(axis=axes).reshape(self.shape)
                self.grad += grad_broad
            if other.requires_grad:
                if other.grad is None:
                    # ∂L/∂B = Aᵀ · G
                    other.grad = np.zeros_like(other.data)
                grad_broad = self.data.swapaxes(-1,-2) @ out.grad
                if grad_broad.shape != other.shape:
                    # axis = tuple(range(grad_broad.ndim - other.ndim))
                    axes = self.get_broadcast_axes(grad_broad.shape, other.shape)
                    grad_broad = grad_broad.sum(axis=axes).reshape(other.shape)
                other.grad += grad_broad

        out._backward = _backward
        out._parents = [self, other]
        return out
    


    # 对一个matrix在一个方向上进行求和
    def sum(self, axis=None, keepdims=False)->'Tensor': 
        current_axis  = axis
        out_data = np.sum(self.data, axis = current_axis , keepdims = keepdims)
        out = Tensor(out_data, requires_grad= self.requires_grad)
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
            # 将上游的梯度reshape成带有keepdims的形状，保证矩阵的维度
                if keepdims:
                    grad_reshape = out.grad
                else:
                    if current_axis  is None:
                        grad_reshape = np.reshape(out.grad, (1,) * self.ndim)
                        # grad_reshape = np.expand_dims(out.grad, axis=tuple(range(self.ndim)))
                    else:
                        # if isinstance(axes,int):
                        #     axes = (current_axis ,)
                        axes = (current_axis,) if isinstance(current_axis, int) else current_axis
                        grad_reshape = out.grad
                        for ax in axes:
                            grad_reshape = np.expand_dims(grad_reshape,axis = ax)
                            
                    grad_broadcast = np.broadcast_to(grad_reshape,self.shape)
                    self.grad += grad_broadcast
        out._backward = _backward
        out._parents = [self]
        return out
    
    def mean(self, axis=None, keepdims=False)->'Tensor':
        out_data = np.mean(self.data, axis = axis, keepdims = keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data) 
                # n = self.data.size // out_data.size #降维轴上的元素数目
                # grad_broadcast = np.broadcast_to(out_data,self.shape)/n
                # self.grad += grad_broadcast
                if axis is None:
                    count = self.data.size
                else:
                    if  isinstance(axis, int):
                        count = self.data.shape[axis]
                    else:
                        count = np.prod([self.data.shape[ax] for ax in axis])  
                    if keepdims:
                        grad_expanded = out.grad
                    else:
                        if axis is None:
                            grad_expanded = np.reshape(out.grad, (1,)*self.ndim)
                        else:
                            axes = (axis,) if isinstance(axis,int) else axis
                            grad_expanded = out.grad
                            for ax in sorted(axes):
                                grad_expanded = np.expand_dims(grad_expanded, axis=ax)
                    # count = self.data.shape[axis] if isinstance(axis, int) else np.prod([self.data.shape[ax] for ax in axis])
                grad_board = np.broadcast_to(out.grad, self.shape)/count
                self.grad += grad_board
        out._backward = _backward
        out._parents = [self]
        return out
    
    # 变形算子：跑一遍进行还原
    def reshape(self, *new_shape)->'Tensor':
        # 处理-1的情况 ？
        # new_shape = tuple(new_shape)
        out_data = np.reshape(self.data, new_shape)
        out = Tensor(out_data, requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad and self.grad is not None:
                self.grad += np.reshape(out.grad, self.shape)
        out._backward = _backward
        out._parents = [self]
        return out
    
    # 在tensor的边缘插入常数值0,使得尺寸变大
    # 公式：out[i] = constant 当 i 落在 pad 区域；否则 out[i] = x[i - pad_left]
    # 输入张量 x 形状：(d₀, d₁, …, d_{n-1})
    # pad_width：( (left₀, right₀), (left₁, right₁), … )
    # 输出 out 形状：(d₀+left₀+right₀, d₁+left₁+right₁, …)
    # ∂L/∂x[j] = Σ_{i ∈ pad_region(j)} ∂L/∂out[i]
    # ∂L/∂x[j] = ∂L/∂out[j + pad_left]
    """
    x = Tensor([[1, 2], [3, 4]], requires_grad=True)
    y = x.pad(((1, 1), (1, 1)))   # 0 填充一圈
    y.backward(Tensor(np.ones_like(y.data)))  # 上游梯度全 1
    """
    # pad_width = ((1, 1),   # 第 0 轴（行）: 前(上) 1，后(下) 1
    #         (1, 1))   # 第 1 轴（列）: 前(左) 1，后(右) 1
    # 复制算子： 块求和还原
    def pad(self, pad_width, mode='constant', constant_values=0)->'Tensor': 
        out_data = np.pad(self.data, pad_width, mode = mode, constant_values = constant_values)
        out = Tensor(out_data, requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad and self.grad is not None:
                # 学一下tuple和slice的用法
                slices = tuple(slice(l,-r if r!= 0 else None) for l,r in pad_width)
                grad_crop = out.grad[slices]
                self.grad += grad_crop
        out._backward = _backward
        out._parents = [self]
        return out
    # 维度交换，交换两个轴的顺序
    # 公式：out[i, j] = x[j, i]
    # 变形算子：跑一遍进行还原
    # def transpose(self, axis1, axis2)->'Tensor':
    #     out_data = np.transpose(self.data, axes = (axis1,axis2))
    #     out = Tensor(out_data,requires_grad=True)
    #     def _backward():
    #         if self.grad is not None:
    #             grad_trans = np.transpose(out.grad, axes = (axis1,axis2))
    #             self.grad += grad_trans
    #     out._backward = _backward
    #     out._parents = [self]
    #     return out


    # 支持多轴交换
    def transpose(self,*axes)->'Tensor':
        if not axes:
            #默认转置所有的轴
            axes = tuple(reversed(range(self.ndim)))
        out_data = np.transpose(self.data, axes)
        out = Tensor(out_data, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad and self.grad is not None:
                # 计算逆转质轴
                inv_axes = np.argsort(axes)
                self.grad += np.transpose(out.grad ,inv_axes)
        out._backward = _backward
        out._parents = [self]
        return out

    # 通过切片获得子张量
    def __getitem__(self, key)->'Tensor':
        out_data = self.data[key]
        out = Tensor(out_data,requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad and self.grad is not None:
                np.add.at(self.grad, key, out.grad)
        out._backward = _backward
        out._parents = [self]
        return out
    
    # 按照索引对Tensor
    def slice(self, key)->'Tensor':
        out_data = self.data[key]
        out = Tensor(out_data, requires_grad=True)
        def _backward():
            if self.grad is not None:
                np.add.at(self.grad, key, out.grad)
        out._backward = _backward
        out._parents = [self]
        return out

    # 在指定的位置插入一个长度为1的维度
    # 变形算子：跑一遍进行还原
    def expand_dims(self, axis)->'Tensor':
        out_data = np.expand_dims(self.data, axis)
        out = Tensor(out_data,requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad and self.grad is not None:
                grad_expand = np.squeeze(out.grad, axis=axis)
                self.grad += grad_expand
        out._backward = _backward
        out._parents = [self]
        return out       

    """
    repeats: int 或 list[int]，每轴重复次数
    axis: 沿哪根轴重复；None 表示扁平后重复
    """
    # 复制算子： 块求和还原
    def repeat(self, repeats, axis)->'Tensor':
        repeats = np.array(repeats, dtype = int)
        out_data = np.repeat(self.data, repeats, axis=axis)
        out = Tensor(out_data,requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad and self.grad is not None:
                # 构造每个块的索引
                indices = np.concatenate([[0],np.cumsum(repeats)])
                # 沿着指定的轴对梯度块求和
                grad_repeat = np.add.reduceat(out.grad, indices[:-1], axis=axis)
                self.grad += grad_repeat
        out._backward = _backward
        out._parents = [self]
        return out   

    def var(self, axis=None, keepdims=False, correction=1):
        """
        计算张量的方差
        
        参数:
            axis: 沿指定轴计算方差，None 表示计算所有元素的方差
            keepdims: 是否保持维度
            correction: 自由度修正，默认为 1 (样本方差)
                       0 表示总体方差，1 表示无偏估计的样本方差
        
        返回:
            方差张量
        """
        # 计算均值
        mean_value = self.mean(axis=axis,keepdims=True)
        # 计算平方偏差
        squarred_diff = (self-mean_value)**2

        # 计算平方偏差的均值
        if axis is None:
            # 所有元素的方差
            n_elements = np.prod(self.shape)
            variance = squarred_diff.sum()/(n_elements - correction)
        else:
            # 指定轴的方差
            if isinstance(axis, int):
                axis = (axis, ) # 将单个轴转化为元组，统一处理
            n_elements = 1
            for ax in axis:
                n_elements *= self.shape[ax] # 计算沿着指定轴的元素总数
            variance = squarred_diff.sum(axis = axis, keepdims = keepdims)/(n_elements - correction)
        return variance


    
    def std(self, axis=None, keepdims=False, correction=1):
        """
        计算张量的标准差
        
        参数:
            axis: 沿指定轴计算标准差，None 表示计算所有元素的标准差
            keepdims: 是否保持维度
            correction: 自由度修正，默认为 1 (样本标准差)
        
        返回:
            标准差张量
        """
        variance = self.var(axis=axis, keepdims=keepdims, correction=correction)
        return variance ** 0.5

    def zero_grad(self):
        if self.grad is not None:
            self.grad.fill(0.0) #原地清零，比创建一个数组更加的高效
        return self

    def max(self, axis=None, keepdims=False) -> 'Tensor':
        current_axis = axis
        out_data = np.max(self.data, axis=current_axis, keepdims=keepdims)
        argmax_val = np.argmax(self.data, axis=current_axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad)
        
        def _backward():
            nonlocal argmax_val, current_axis
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                
                grad_mask = np.zeros_like(self.data)
                
                if current_axis is None:
                    # 全局最大值
                    grad_mask.flat[argmax_val] = out.grad
                else:
                    if isinstance(current_axis, (tuple,list)):
                        raise NotImplementedError("not support multiply axis min, by yrq, in core min")
                    
                    # 手动实现put_along_axis的逻辑
                    if keepdims:
                        indices = argmax_val
                    else:
                        # 为argmax_val添加被压缩的维度
                        indices = np.expand_dims(argmax_val, axis=current_axis)
                    
                    # 手动创建索引并赋值
                    indices_tuple = []
                    for i in range(self.data.ndim):
                        if i == current_axis:
                            indices_tuple.append(indices)
                        else:
                            # 创建广播索引
                            idx = np.arange(self.data.shape[i])
                            shape = [1] * self.data.ndim
                            shape[i] = self.data.shape[i]
                            idx = idx.reshape(shape)
                            indices_tuple.append(idx)
                    
                    # 确保out.grad的形状与索引选择的形状匹配
                    selected_shape = indices_tuple[0].shape
                    if out.grad.size == 1:
                        # 标量情况
                        grad_value = out.grad
                    else:
                        # 调整out.grad的形状
                        grad_reshaped = out.grad.reshape(selected_shape)
                        grad_value = grad_reshaped
                    
                    # 使用高级索引赋值
                    grad_mask[tuple(indices_tuple)] = grad_value
                
                self.grad += grad_mask
        
        out._backward = _backward
        out._parents = [self]
        return out

    def min(self, axis=None, keepdims=False) -> 'Tensor':
            current_axis = axis
            out_data = np.min(self.data, axis=current_axis, keepdims=keepdims)
            argmin_val = np.argmin(self.data, axis=current_axis, keepdims=keepdims)
            out = Tensor(out_data, requires_grad=self.requires_grad)
            
            def _backward():
                nonlocal argmin_val, current_axis
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = np.zeros_like(self.data)
                    
                    grad_mask = np.zeros_like(self.data)
                    
                    if current_axis is None:
                        # 全局最小值
                        grad_mask.flat[argmin_val] = out.grad
                    else:
                        if isinstance(current_axis, (tuple,list)):
                            raise NotImplementedError("not support multiply axis min, by yrq, in core min")
                        # 手动实现put_along_axis的逻辑
                        if keepdims:
                            indices = argmin_val
                        else:
                            # 为argmin_val添加被压缩的维度
                            indices = np.expand_dims(argmin_val, axis=current_axis)
                        
                        # 手动创建索引并赋值
                        indices_tuple = []
                        for i in range(self.data.ndim):
                            if i == current_axis:
                                indices_tuple.append(indices)
                            else:
                                # 创建广播索引
                                idx = np.arange(self.data.shape[i])
                                shape = [1] * self.data.ndim
                                shape[i] = self.data.shape[i]
                                idx = idx.reshape(shape)
                                indices_tuple.append(idx)
                        
                        # 确保out.grad的形状与索引选择的形状匹配
                        selected_shape = indices_tuple[0].shape
                        if out.grad.size == 1:
                            # 标量情况
                            grad_value = out.grad
                        else:
                            # 调整out.grad的形状
                            grad_reshaped = out.grad.reshape(selected_shape)
                            grad_value = grad_reshaped
                        
                        # 使用高级索引赋值
                        grad_mask[tuple(indices_tuple)] = grad_value
                    
                    self.grad += grad_mask
            
            out._backward = _backward
            out._parents = [self]
            return out

    @property
    def T(self)->'Tensor':
        if self.ndim !=2:
            raise ValueError(f"T only used in 2 dim, the current dim is {self.ndim}")
        return self.transpose(1,0)

    # 是python的内置的方法,返回对象占用的内存字节数，和元素个数无关
    # @property
    # def __sizeof__(self):
    #     return self.data.size
    
    @property
    def size(self):
        return self.data.size
    
    # ===== 反向传播入口 =====
    def backward(self, grad_output=None): 
    # 内部：拓扑排序 + 链式回调
        
        if grad_output is None:
            # 若当前张量是标量（shape=()），默认梯度为 1.0
            if self.data.ndim == 0:
                # grad_output = np.ones_like(self.data)
                grad_output = np.array(1.0,dtype = np.float32)
            else:
                raise ValueError("grad_output must be provided for non-scalar tensors, written by yrq, in backward")
        else:
            # 确保grad_output 是numpy数组且形状匹配
            grad_output = np.asarray(grad_output, dtype=np.float32)
            if grad_output.shape!=self.shape:
                raise ValueError(f"the shape {grad_output.shape} and shape {self.shape} is not mathch, written by yrq, in backward")
        topo = []
        visited = set()
        # 拓扑排序（DFS逆序）
        def build_topo(v):
            if id(v) not in visited:
                visited.add(id(v))
                for parent in v._parents:
                    build_topo(parent)
                topo.append(v)
        build_topo(self)

        # 存在0-D标量不能使用[:]赋值
        # if self.grad is None:
        #     self.grad = np.zeros_like(self.data)
        # self.grad[:] = grad_output #使用切片赋值保证形状
        if self.grad is None or self.grad.shape != self.data.shape:
            self.grad = np.empty_like(self.data) # 首次或者形状变化的时候重新分配
        np.copyto(self.grad, grad_output) #支持0-D -> N-D广播

        for node in reversed(topo):
            node._backward()
        return self
    
# 测试函数
if __name__ == "__main__":
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