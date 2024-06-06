
## nan/inf detect

用户只需在初始化模型后，调用 hook 注入功能。继续运行中如果出现 nan/inf，将会直接返回故障模块的调用堆栈

```python
import torch
import torch.nn as nn

from dl_debugger.assert_value import check_model_forward_infinite


model = MTransformer(64, 512)
# insert hooks
check_model_forward_infinite(model)

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
