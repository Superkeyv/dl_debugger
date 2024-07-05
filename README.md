快速定位神经网络的异常

本工具通过附加 hook 的方式运行，不需要侵入用户的源代码，易于与任何 pytorch 框架实现的模型相结合

实现的功能包括：

- fwd/bwd 参数分析
- nan/inf 异常值检测
- tensor hash 值计算


# install

```bash
python setup.py install
```

# usage

## nan, inf value detect

```python
# 获取 异常值检测器
from dl_debugger.assert_value import register_check_model_fwd_nan

model = GPT2()
# insert hooks
register_check_model_fwd_nan(model)
```

接下来可以正常执行模型训练，如果检测到 nan/inf 数值，则会立即打印堆栈信息，方便用户确定 nan/inf 的产生位置


更多内容参考 [invalid value check](docs/invalid%20value%20check.md)

## fwd, bwd param analyze

更多内容参考 [forward backward check](docs/forward%20backward%20check.md)

