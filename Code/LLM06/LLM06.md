# TODO

- [ ] 弄明白`BufferCache`
- [ ] 准确性检查（有兴趣再看）
  - [ ] 关于`effective_n_kv_heads`
- [ ] 计算效率检查（有兴趣再看）
  - [ ] 关于`enable_flash_sdp`
  - [ ] 关于`enable_mem_efficient_sdp`
  - [ ] 关于`ActivationCheckpointingStrategy.fine_grained`




## model.py

### 代码结构

下面列表展示父类和子类之间的关系及功能。

- `LayerNormBase`：父类，根据参数设定是否使用权重、低精度和RMSNorm

  - `LayerNorm`：根据设定好的参数，调用`F.layer_norm`完成实现

  - `RMSLayerNorm`：按照RMSNorm的公式进行计算，$y_i=\frac{x_i}{\mathrm{RMS}(x)}*\gamma_i,\mathrm{RMS}(x)=\sqrt{\epsilon+\frac1n\sum_{i=1}^nx_i^2}$

- `Activation`：父类，根据传入的参数返回对应的激活函数类
  - `GELU`

  - `ReLU`

  - `SwiGLU`

- `RotaryEmbedding`：现在常用的token位置编码方法[RoPE](https://spaces.ac.cn/archives/8265/comment-page-1)的实现，融合了绝对位置编码和相对位置编码

- `OLMoBlock`：构成transformer模型的基本块



|   属性    |                赋值                | 作用 |
| :-------: | :--------------------------------: | :--: |
| `dropout` | `Dropout(config.residual_dropout)` |      |
| `k_norm`  |                                    |      |
| `q_norm`  |                                    |      |



### 加速技巧？

```python
# 禁用或启用FlashAttention
torch.backends.cuda.enable_flash_sdp(True)  
# 启用或禁用 Memory-Efficient Attention（原注释：非常慢，不要开）
torch.backends.cuda.enable_mem_efficient_sdp(False)  
# 还没有弄明白
BufferCache
```

