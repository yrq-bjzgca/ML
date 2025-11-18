
from typing import Dict, Iterator, Optional, Tuple, Callable, List  # 添加 

import sys
sys.path.append("..")
from core.tensor import Tensor

class Module:
    """
    神经网络模块基类
    所有神经网络模块都应该继承这个类
    """
    
    def __init__(self):
        """初始化模块"""
        # self._modules: Dict[str, Module] = {}
        # self._parameters: Dict[str, Tensor] = {}
        # self.training = True  # 默认处于训练模式

        # 使用object._setattr_ 避免误触发_setattr_
        object.__setattr__(self, '_modules', {})
        object.__setattr__(self, '_parameters', {})
        object.__setattr__(self, 'training', True)
        object.__setattr__(self, '_forward_hooks', []) 
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量
            
        返回:
            输出张量
        """
        raise NotImplementedError("子类必须实现forward方法")
    
    def __call__(self, x: Tensor) -> Tensor:
        """使实例可调用"""
        return self.forward(x)
    
    def parameters(self) -> Iterator[Tensor]:
        """
        返回模块的所有参数
        
        返回:
            参数迭代器
        """
        # TODO: 返回所有可训练参数
        # 包括当前模块的参数和所有子模块的参数

        # 收集当前模块的函数
        # for name,param in self._parameters.items():
        # for param in object.__getattribute__(self, '_parameters').values():
            # yield param

        # 收集所有的子模块的函数
        # for name,module in self._modules.items():
        # for module in object.__getattribute__(self, '_modules').value:
            # for param in module.parameters():
            #     yield param
            # yield from module.parameters()
        _parameters = object.__getattribute__(self, '_parameters')
        _modules = object.__getattribute__(self,'_modules')

        # 收集当前模块的参数
        for param in _parameters.values():
            yield param

        # 收集所有的子模块的参数
        for module in _modules.values():
            yield from module.parameters()


    def named_parameters(self, prefix: str = '') -> Iterator[tuple]:
        """
        返回模块的所有可训练参数及其名称
        
        参数:
            prefix: 名称前缀
            
        返回:
            (名称, 参数) 元组的迭代器
        """
        # 收集当前模块
        _parameters = object.__getattribute__(self, '_parameters')
        _modules = object.__getattribute__(self,'_modules')

        for name,param in _parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, param

        # 收集所有子模块的函数
        for name,module in _modules.items():
            full_prefix = f"{prefix}.{name}" if prefix else name
            for param_name, param in module.named_parameters(full_prefix):
                yield param_name, param

    def children(self) -> Iterator['Module']:
        """
        返回直接子模块迭代器
        
        返回:
            子模块迭代器
        """
        # TODO: 返回直接子模块迭代器
        yield self
        _modules = object.__getattribute__(self, '_modules')
        for module in _modules.values():
            yield module

    def modules(self) -> Iterator['Module']:
        """
        返回所有模块的迭代器（包括自身）
        
        返回:
            模块迭代器
        """
        # TODO: 返回所有模块的迭代器
        yield self
        _modules = object.__getattribute__(self, '_modules')
        # for name, module in self._modules.items():
        #     for m in module.modules():
        #         yield m
        for module in _modules.values():
            yield from module.modules()

    def add_module(self, name: str, module: 'Module') -> None:
        """
        添加子模块
        
        参数:
            name: 子模块名称
            module: 子模块实例
        """
        # TODO: 添加子模块到_modules字典
        _modules = object.__getattribute__(self, '_modules')
        if module is None:
            if name in _modules:
                del _modules[name]
            # self._modules.pop(name, None)
        else:
            self._modules[name] = module
    

    def train(self) -> None:
        """设置为训练模式"""
        # TODO: 设置当前模块和所有子模块为训练模式

        # self.training =True
        object.__setattr__(self,'training',True)
        _modules = object.__getattribute__(self, '_modules')
        for module in _modules.values():
            module.train()
    
    def eval(self) -> None:
        """设置为评估模式"""
        # TODO: 设置当前模块和所有子模块为评估模式
        object.__setattr__(self,'training',False)
        _modules = object.__getattribute__(self, '_modules')
        for module in _modules.values():
            module.eval()
    
    def zero_grad(self) -> None:
        """清零所有参数的梯度"""
        # TODO: 遍历所有参数，将梯度置零
        for param in self.parameters():
            if param.grad is not None:
                param.grad.fill(0.0)
   
    def register_forward_hook(self, hook: Callable[['Module', Tuple[Tensor, ...], Tensor], Optional[Tensor]]) -> Callable:
        """
        注册前向传播hook
        
        参数:
            hook: 钩子函数，签名为 hook(module, input, output) -> None 或新output
                - module: 当前模块
                - input: 输入元组 (Tensor,)
                - output: 模块输出
        
        返回:
            钩子函数本身，可用于后续移除
        """
        _forward_hooks = object.__getattribute__(self, '_forward_hooks')
        _forward_hooks.append(hook)
        return hook

    def remove_forward_hook(self, hook: Callable) -> None:
        """
        移除已注册的前向hook
        
        参数:
            hook: 要移除的钩子函数
        """
        _forward_hook = object.__getattribute__(self, '_forward_hooks')
        if hook in _forward_hook:
            _forward_hook.remove(hook)
        else:
            raise ValueError("Hook not found in the module's forward hooks")

    def __setattr__(self, name: str, value) -> None:
        """
        设置属性
        
        参数:
            name: 属性名
            value: 属性值
        """
        # TODO: 特殊处理模块和参数的设置
        # 如果value是Module实例，添加到_modules
        # 如果value是Tensor且requires_grad=True，添加到_parameters
        # 否则正常设置属性
        # 获得当前的_modules 和 _parameters
        _modules = object.__getattribute__(self,'_modules')
        _parameters = object.__getattribute__(self,'_parameters')
        if isinstance(value, Module):
            _modules[name] = value
        elif isinstance(value, Tensor) and value.requires_grad:
            _parameters[name] = value
        # 正常设置属性
        else:
            object.__setattr__(self, name, value)

    def __call__(self, x:Tensor)->Tensor:
        """实例可以调用"""
        output = self.forward(x)
        # 触发向前hook
        _forward_hooks = object.__getattribute__(self, '_forward_hooks')
        for hook in _forward_hooks:
            result = hook(self,(x),output)
            if result is not None:
                output = result
        return output

    def __getattr__(self, name: str):
        """
        获取属性
        
        参数:
            name: 属性名
            
        返回:
            属性值
        """
        # TODO: 特殊处理模块和参数的获取
        # 如果在_modules中，返回对应模块
        # 如果在_parameters中，返回对应参数
        # 否则抛出AttributeError

        # 获得当前的_modules 和 _parameters
        _modules = object.__getattribute__(self,'_modules')
        _parameters = object.__getattribute__(self,'_parameters')

        if name in self._modules:
            # return self.modules[name]
            return _modules[name]
        elif name in self._parameters:
            # return self._parameters[name]
            return _parameters[name]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no "\
                                 "attribute '{name}'")

    def __repr__(self) -> str:
        """
        返回模块的字符串表示
        """
        return f"{self.__class__.__name__}()"
    
    def extra_repr(self) -> str:
        """
        返回额外的描述信息，子类可以重写此方法
        
        返回:
            额外的描述字符串
        """
        return ""
    
    def _apply(self, fn) -> 'Module':
        """
        应用函数到所有参数和子模块
        
        参数:
            fn: 要应用的函数
            
        返回:
            self
        """
        _parameters = object.__getattribute__(self, '_parameters')
        _modules = object.__getattribute__(self, '_modules')
        # 应用到参数
        for param in _parameters.values():
            # 可以移动到其他的设备中
            pass
        # 应用到子模块
        for module in _modules.values():
            module._apply(fn)
        return self

    def state_dict(self, prefix:str) -> Dict:
        """
        返回模块的状态字典
        参数:
        prefix: 键的前缀
        返回:
            状态字典
        """
        state_dict = {}
        _parameters = object.__getattribute__(self, '_parameters')
        _modules = object.__getattribute__(self, '_modules')

        # 收集参数
        for name, param in _parameters.items():
            full_nam = f"{prefix}.{name}" if prefix else name
            state_dict[full_nam] = param.data.copy()

        # 收集子模块的状态
        for name, module in _modules.items():
            full_prefix = f"{prefix}.{name}" if prefix else name
            module_state = module.state_dict(full_prefix)
            state_dict.update(module_state)
            # if module_state:
                # state_dict[name] = module_state
            
        return state_dict

    def load_state_dict(self, state_dict: Dict, strict: bool = True) -> None:
        """
        加载状态字典
        
        参数:
            state_dict: 状态字典
            strict: 是否严格匹配键值
        """
        _parameters = object.__getattribute__(self, '_parameters')
        _modules = object.__getattribute__(self, '_modules')

        missing_keys = []
        unexpected_keys = list(state_dict.keys())

        # 加载当前的模块的参数
        for name, param in _parameters.items():
            if name in state_dict:
                param.data = state_dict[name]
                unexpected_keys.remove(name)
            else:
                missing_keys.append(name)

        # 递归加载子模块的状态
        for name, module in _modules.items():
            # 构建子模块的参数前缀
            module_prefix = f"{name}."
            module_state_dict = {}
            for key in list(state_dict.keys()):
                if key.startwith(module_prefix):
                    #移除前缀，得到子模块的内部的键
                    sub_key = key[len(module_prefix):]
                    module_state_dict[sub_key] = state_dict[key]
                    unexpected_keys.remove(key)
            # if name in state_dict:
            #     module.load_state_dict(state_dict[name], strict) #递归调用
            #     unexpected_keys.remove(name)
            # else:
            #     missing_keys.append(f"{name}.*")
        
        # 严格模式下的错误的处理
        if strict:
            if missing_keys:
                raise RuntimeError(f"missing : {missing_keys}")
            if unexpected_keys:
                raise RuntimeError(f"unexpect : {unexpected_keys}")
    def named_modules(self, prefix: str = '') -> Iterator[Tuple[str, 'Module']]:
        """
        返回所有模块的迭代器（包括自身），包含名称前缀
        
        参数:
            prefix: 名称前缀
            
        返回:
            (名称, 模块) 元组的迭代器
        """
        # 首先返回自身
        yield prefix, self
        # 递归遍历所有子块
        _modules = object.__getattribute__(self, '_modules')
        for name, module in _modules.items():
            # 构建子模块的完全名称路径
            if prefix:
                submodule_prefix = f"{prefix}.{name}"
            else:
                submodule_prefix = f"layer_{name}_{type(module).__name__}"
            # 递归获得子模块的所有嵌套模块
            yield from module.named_modules(submodule_prefix)