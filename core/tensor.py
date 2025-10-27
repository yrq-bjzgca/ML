import numpy as np
from typing import List, Optional, Union, Tuple

class Tensor:
    """
    手写可微张量
    要求：支持广播、切片、pad；记录计算图；链式反向
    """
    def __init__(self, data, requires_grad=False):
        self.data = np.asarray(data, dtype=np.float32)
        self.shape = self.data.shape
        self.grad = None
        if requires_grad:
            self.grad = np.zeros_like(self.data)
        self._backward = lambda: None #反向函数
        self._parents = [] #计算图父亲
    # ---------- 工具 ----------
    def __repr__(self):
        return f"Tensor({self.data}, shape={self.shape}, requires_grad={self.grad is not None})"
    
    # ===== 基础算子（留空实现） =====
    def __add__(self, other:'Tensor')->'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        # 基础版本
        # out = Tensor(self.data + other.data, requires_grad=True)
        # # 整理计算图
        # def _backward(): 
        #     if self.grad is not None:
        #         self.grad += out.grad
        #     if other.grad is not None:
        #         other.grad += out.grad
        # out._backward = _backward # 将函数当回存调存起来，只有当函数最终backward的时候才会执行
        # out._parents = [self, other]
        # return out

        # 带有广播
        a,b  = np.boardcat_arrsys(self.data, other.data)
        out = Tensor(a + b, requires_grad=True)
        def _backward():
            if self.grad is not None:
                grad_boardcast = out.grad
                axis = tuple(range(grad_boardcast.ndim - self.ndim))
                self.grad += grad_boardcast.sum(axis=axis).reshape(self.shape)
            if other.grad is not None:
                grad_boardcast = out.grad
                axis = tuple(range(grad_boardcast.ndim - self.ndim))
                self.grad += grad_boardcast.sum(axis=axis).reshape(self.shape)
            
        out._backward = _backward()
        out._parents = [self, other]
        return out

    def __mul__(self, other:'Tensor')->'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
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
        a, b = np.boardcast_arrays(self.data, other.data)
        out = Tensor(a * b, requires_grad= True)
        def _backward():
            if self.grad is not None:
                grad_board = out.grad * b
                axis = tuple(range(grad_board.ndim - self.ndim))
                self.grad += grad_board.sum(axis=axis).reshape(self.shape)
            if other.grad is not None:
                grad_board = out.grad * a
                axis = tuple(range(grad_board - other.ndim))
                other.grad += grad_board.sum(axis=axis).reshape(other.shape)
        out._backward = _backward
        out._parents = [self, other]
        return out
    
    def exp(self:'Tensor', )->'Tensor':...

    def log(self:'Tensor', )->'Tensor':...


    def __matmul__(self, other:'Tensor')->'Tensor':...


 
    def sum(self, axis=None, keepdims=False): ...

    def mean(self, axis=None, keepdims=False): ...

    def reshape(self, *new_shape): ...

    def pad(self, pad_width, mode='constant', constant_values=0): ...

    def transpose(self, ):...

    def slice(self, ):...

    def expand_dims(self, ):...

    def repeat(self, ):...


        
    # ===== 反向传播入口 =====
    def backward(self, grad_output=None): ...
    # 内部：拓扑排序 + 链式回调