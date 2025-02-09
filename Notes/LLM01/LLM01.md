# Notes

## 胡言乱语

本文主要是作者的学习笔记，源头是希望自己有一定的知识储备和对LLM感兴趣（其实是导师的项目没有指导不太做得下去，顺带追下热点）。

所以希望以某个LLM教程作为基础，自己选择性的学习教程中的知识，同时会根据学习的进展自由补充拓展相关的知识内容，尽量记录下学习过程中参考过的所有资料进行引用，这样在简略带过的地方也可以给想要了解的同学一个方向，希望最后可以形成一个系统、完整的学习路径帮助到更多想要学习LLM的同学，希望大家一起学习一起进步！

学习过程中，默认有一定的深度学习基础，了解基础的神经网络、反向传播、CV和NLP的知识，为了不造成混淆和困惑，大部分术语第一次出现时保留英文。虽然文章大部分都是关于LLM的内容，但是很多内容比如分布式训练的部分都可以在其他邻域的模型上使用。另外，一些在实际中基本不会用到代码的就不作代码记录了，一些比较常用的可能每次写代码都要看一下来补充的就会进行记录。

现在第一篇就简单做一下LLM的基础引入吧！


## 主线

[https://github.com/mlabonne/llm-course](https://github.com/mlabonne/llm-course)

# **The LLM architecture(LLM架构)**

## Attention(注意力机制)

代码来源和对应的解说（视频娓娓道来讲的非常好，强烈建议学习）：

[Google Colab](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing)

[Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)

实现代码

```python
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        # config is a dataclass included n_embd, n_head and dropout
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # x: (B, T, C)
        # (B, T, nh * hs) -> (B, T, nh, hs) -> (B, nh, T, hs)
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)  
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            # Efficiently compute scaled dot-product attention using PyTorch's built-in function.
            # This function leverages Flash Attention CUDA kernels for optimized performance.
            # Args: 
            #     q: Query tensor of shape (B, nh, T, hs), where B is batch size, nh is number of heads,
            #        T is sequence length, and hs is head size.
            #     k: Key tensor of shape (B, nh, T, hs).
            #     v: Value tensor of shape (B, nh, T, hs).
            #     attn_mask: Optional attention mask. Set to None since causal masking is handled internally.
            #     dropout_p: Dropout probability applied to attention weights during training.
            #     is_causal: If True, applies a causal mask to ensure attention is only applied to the left in the sequence.
            # Returns:
            #     y: Output tensor of shape (B, nh, T, hs) after applying attention.
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, 
                                                                 attn_mask=None, 
                                                                 dropout_p=self.dropout if self.training else 0, 
                                                                 is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
```

其中进行注意力mask计算部分的原理为

```python
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))  # 将tril中为0的位置替换为负无穷
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
torch.allclose(xbow, xbow3)
```

## **Decoding Strategies**(解码策略)

简略记录，更详细的内容和代码展示在：

[Maxime Labonne - Decoding Strategies in Large Language Models](https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html)

### **Greedy Search**(贪婪搜索)

每一步选择概率最高的token作为序列中的下一个token。

虽然这种方法听起来很直观，但需要注意的是，贪婪搜索是短视的：它只考虑每一步最可能的token，而不考虑对序列的整体影响。这一特性使其快速高效，因为它不需要跟踪多个序列，但也意味着它可能会错过那些在下一步选择稍低概率token时可能出现的更好序列。

### **Beam Search**(束搜索)

与贪婪搜索只考虑下一个最可能的token不同，束搜索会考虑n个最可能的token，其中n表示束的数量。这个过程会重复进行，直到达到预定义的最大长度或出现序列结束token。此时，选择总体得分最高的序列（或“束”）作为输出。

我们可以调整之前的函数，使其考虑n个最可能的token，而不仅仅是一个。在这里，我们将维护序列得分logP(w)，这是束中每个token的对数概率的累积和。我们通过序列长度对该得分进行归一化，以防止对较长序列的偏见（此因子可以调整）。再次，我们将生成五个额外的token来完成句子“I have a dream.”。

### **Top-k sampling**(Top-k采样)

Top-k采样是一种利用语言模型生成的概率分布从k个最可能的选项中随机选择token的技术。

举例来说，假设我们有k=3和四个token：A、B、C和D，其概率分别为：P(A)=30，P(B)=15，P(C)=5，P(D)=1。在top-k采样中，token D被忽略，算法将在60%的时间内输出A，30%的时间内输出B，10%的时间内输出C。这种方法确保我们优先考虑最可能的token，同时在选择过程中引入随机性。

引入随机性的另一种方法是**温度（temperature）**的概念。温度T是一个范围从0到1的参数，它影响softmax函数生成的概率，使最可能的token更具影响力。在实践中，它只是将输入的logits除以一个我们称为温度的值：

$$
\mathrm{softmax}(x_i)=\frac{e^{x_i/T}}{\sum_je^{x_j/T}}
$$

温度为1.0相当于没有温度的默认softmax（如Deepseek的api文档中就有提到，默认的temperature为1）。另一方面，低温设置（0.1）会显著改变概率分布。这通常用于文本生成中，以控制生成输出的“创造力”水平。通过调整温度，我们可以影响模型生成更多样化或更可预测的响应的程度。（温度越高，生成的文本越随机；温度越低，生成的文本越固定）

### **Nucleus sampling**(核采样)

核采样，也称为top-p采样，与top-k采样采取了不同的方法。它不是选择前k个最可能的token，而是选择一个截止值p，使得所选token的概率之和超过p。这形成了一个“核”，从中随机选择下一个token。

换句话说，模型按降序检查其最可能的token，并不断将它们添加到列表中，直到总概率超过阈值p。与top-k采样不同，核中包含的token数量可能每一步都不同。这种可变性通常会导致更多样化和创造性的输出，使得核采样在文本生成等任务中很受欢迎。

## GPU Memory Usage Calculation(显存计算)

[Calculating GPU memory for serving LLMs | Substratus.AI](https://www.substratus.ai/blog/calculating-gpu-memory-for-llm)

### 推理显存计算

$$
M=\frac{(P*4B)}{(32/Q)}*1.2
$$

| Symbol | Description |
| --- | --- |
| M | GPU显存以吉字节（Gigabyte）表示 |
| P | 模型的参数量（单位B）。例如，7B模型表示具有70亿参数(parameters) |
| 4B | 4 bytes，表示每个参数(parameter)所占用的字节(bytes)数 |
| 32 | 4 bytes 包含 32 bits |
| Q | 模型加载时应使用的位宽(bits)。例如：16 bits、8 bits 或 4 bits |
| 1.2 | 表示GPU显存中加载额外内容所需的20%额外开销（overhead） |

按照这个公式计算出来的单位是GB，如果不乘以1.2得到的就是单个模型所占的显存。

### 训练显存计算

如果需要进行训练，根据下面这篇文章的内容，如果是进行全量训练的话，显存的占用量一般是推理的4倍。

[根据LLM参数量估算显存/内存占用](https://cuiyuhao.com/posts/c87c0f5d/)

### 参数量计算代码实现

```python
# Code From Deepseek-R1
def get_model_info(model, print_info=True):
    """获取模型关键参数信息
    
    Args:
        model: torch.nn.Module
        print_info (bool): 是否打印信息
    
    Returns:
        dict: {
            'total_params': 总参数量,
            'trainable_params': 可训练参数量,
            'dtypes': 参数数据类型集合
        }
    """
    # 计算参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 获取数据类型
    param_dtypes = {p.dtype for p in model.parameters()}
    
    # 整理结果
    info_dict = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'dtypes': param_dtypes
    }
    
    # 打印信息（按需）
    if print_info:
        print(f"[模型参数信息]")
        # 关键修改点：添加B(十亿)单位显示
        print(f"总参数量: {total_params:,} ({total_params / 1e9:.2f}B)")
        print(f"可训练参数: {trainable_params:,} ({trainable_params / 1e9:.2f}B)")
        print(f"参数类型: {param_dtypes}")
    
    return info_dict

# 使用示例 ------------------------------------------------------------
# model = YourModelClass()
# model_info = get_model_info(model)
```
