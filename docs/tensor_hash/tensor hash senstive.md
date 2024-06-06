我们将噪声和 tensor 进行比较，分别计算原始数据的 hash，以及加噪数据的 hash,比较相对损失，可以得到如下对应关系

![](tensor_hash_cmp_torch.float32.png)
![](tensor_hash_cmp_torch.bfloat16.png)