前向，反向传播检查

该工具可以对模型每个 module 的输入和输出进行分析，辅助开发者确定同一个模型的多种实现是否存在差异

# 核心功能

1. 提供多个级别的 tensor 描述，包括 tensor_hash (用一个 float 数来描述tensor), tensor_fingerprint (用一组 float 数描述tensor), tensor_raw (保存原始 tensor, 以备后续分析)
    1. 默认使用 tensor_hash
    2. 配置 `DLSAN_TENSOR_DETAIL=true`，以使用 tensor fingerprint 描述 tensor
    3. 如果需要保存原始 tensor，需要配置 `DLSAN_DUMP_RAW_TENSOR=true`，此时不会记录 tensor_hash 和 tensor_fingerprint
    4. 如果需要在命令行显示 tensor_hash 或 tensor_fingerprint, 需要配置 `DLSAN_LOG_LEVEL=debug`
2. 逐 module 的 tensor 描述保存 (由 dump_file.py 提供)，自动完成
    1. 用户需要配置 `DLSAN_DUMP=<path>` 环境变量，以决定抓取的 tensor 描述文件(parquet 或 pickle 格式) 存放路径，默认是当前目录下的 `dlsan_dump` 文件夹


# 使用示例

```python
import torch
import torch.nn as nn

from dl_debugger.autograd_debugger import register_fwd_hook, register_bwd_hook


model = MTransformer(64, 512)

# 注册前向、反向传播 hook，这里使用正则表达式过滤要注册的模块
# skip_act_recomp 参数，用于兼容激活重算，避免重复记录模块输出
pattern = r"*.encoder.*.attention.*"
register_fwd_hook(model, pattern=pattern, skip_act_recomp=True)
register_bwd_hook(model, pattern=pattern)


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.L1Loss()

# run the model
epoch = 1
test_x = torch.randn((128, 12, 64))
test_y = torch.randn((128, 12, 64))
for i in range(epoch):
    optimizer.zero_grad()

    out = model(test_x)
    loss = loss_fn(out, test_y)
    loss.backward()
    optimizer.step()

    print(f' epoch [{i}]: current loss {loss.item()}')
```
