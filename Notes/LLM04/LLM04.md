> 希望尽可能将自己学习过程中参考过的资料进行系统的整理，方便后面的初学者更快的找到自己想要的资料！

**笔记持续更新中......**

[LLM基础学习01：LLM解码策略和显存占用计算](https://zhuanlan.zhihu.com/p/21348048780)

[LLM基础学习02：分布式训练核心架构与多级并行策略详解——DDP/FSDP/ZeRO实战代码、显存优化方案及技术资源全景索引](https://zhuanlan.zhihu.com/p/21784954155)

[LLM基础学习03：Qwen2.5-1.5B-Instruct指令微调全流程实践——LLaMA Factory框架与GSM8K评估](https://zhuanlan.zhihu.com/p/22864281740)

**本文的所有代码都放在了仓库[Basic-LLM-Learning](https://github.com/CYRYGBG/Basic-LLM-Learning)中，欢迎star！！！**

# 简介

根据前一篇最后的结论，现在决定选取选取[Qwen/Qwen2.5-1.5B](https://www.modelscope.cn/models/Qwen/Qwen2.5-1.5B)作为base model，选取[GSM8K](hhttps://huggingface.co/datasets/openai/gsm8k)作为微调数据集完整整篇文章的实验和记录。

本文主要依据[LLaMA Factory](https://llamafactory.readthedocs.io/zh-cn/latest/advanced/adapters.html#)中LoRA相关的部分进行原理的学习和代码实验的比较（包括原始LoRA、LoRA+、rsLoRA、DoRA和PiSSA），并且在每个微调方法中贴上对应的论文链接。后文中**全部方法中与微调相关的参数均由Deepseek的建议设置，其他参数全部相同**，每个方法中的“实验”小节仅展示训练过程的曲线，最终结果在“结果对比”中进行展示。

本文涉及到的代码都是基于[LLM基础学习03](https://zhuanlan.zhihu.com/p/22864281740)修改对应的训练配置文件实现的，所以想要跟着一起跑一遍的同学可能需要回头看一下这个。

*题外话：本来还是想测一下全参微调方面的代码的，可惜沿着之前的模型和数据集做测试的话显存不够用了。。。所以只能测一下LoRA相关的。*

# LoRA

## 原理

**论文链接：**[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

LoRA方法认为，大模型在参数微调过程中，参数的更新不是在全参数空间上进行的，而是在一个维度更低的空间中进行（即更新的参数所构成的矩阵实际上是一个低秩矩阵），所以**模型的优化可以在低维空间进行**，也就是低秩分解矩阵（一个低秩的矩阵可以分解为两个简单矩阵的乘积）中进行。另外的一个重要的优点就是，由于LoRA微调的结果是一些比较小的矩阵，在训练后使用时是直接加到原模型上，这就大大方便了细分方向任务的微调，不用每个任务都存一个巨大的模型，而是只要存一个LoRA微调的结果再叠加到原模型后就可以在细分任务中使用。

现在假设一个超级简单的预训练好的模型里面有一个权重矩阵$W_0$，
$$
h=W_1x=W_0x+\Delta Wx=W_0x+BAx
$$
现在我们要基于这个权重针对具体任务进行微调，微调后的权重是$W_1$，由于$W_1$是在$W_0$的基础上梯度下降逐步更新来的，所以**整个微调过程可以视为是在原权重$W_0$上进行加加减减的操作**，我们把整个微调过程中所有的这种操作叠加记为$\Delta W$，这个实际上就是微调过程中模型学习到的东西（所谓参数的更新），那么根据前面的结论：“**模型的优化可以在低维空间进行**”，可以进一步使用两个矩阵相乘来表示这个参数更新，即$B$和$A$，可以参考论文中的图示：

![image-20250213170920858](https://gitee.com/fbanhua/figurebed/raw/master/images/20250213170920905.png)

图中$W$表示原权重矩阵，$B$和$A$表示微调过程中学习到的参数，按照图中的维度，原来需要训练的参数量是$d\times d=d^2$个参数，使用矩阵分解后需要训练的参数量就是$2dr$，假如$d=64,r=8$，**最终就是$4096$和$1024$的区别**，差了四倍！而按照论文中提到的在GPT3上的训练，显存的需求直接从1.2TB变成了350GB（虽然还是很大就是了）。

![image-20250213171603004](https://gitee.com/fbanhua/figurebed/raw/master/images/20250213171603050.png)

## 参数

```yaml
finetuning_type: lora  # lora微调
lora_target: all
lora_rank: 8  # 16
flash_attn: fa2
```

## 实验

![image-20250213172611377](https://gitee.com/fbanhua/figurebed/raw/master/images/20250213172611428.png)

![image-20250213172705571](https://gitee.com/fbanhua/figurebed/raw/master/images/20250213172705620.png)

![image-20250214102654304](https://gitee.com/fbanhua/figurebed/raw/master/images/20250214102654398.png)

# LoRA+

## 原理

**论文链接：**[LoRA+: Efficient Low Rank Adaptation of Large Models](https://arxiv.org/abs/2402.12354)

在LoRA中，矩阵$B$使用全0初始化，矩阵$A$使用随机高斯初始化（从高斯分布/正态分布中随机采样初始值）；假如有两种初始化方式如下：

![image-20250213185144751](https://gitee.com/fbanhua/figurebed/raw/master/images/20250213185144798.png)

其中$\sigma_{b}, \sigma_{a}$分别表示使用高斯初始化时$B$和$A$的正态分布的方差取值。如果假设模型为以下公式的简单情况：
$$
f(x)=(W^*+ba^\top)x,
$$
$W^*\in\mathbb{R}^{1\times n}$为预训练后的模型权重，$b\in\mathbb{R},a\in \mathbb{R}^{n}$为对应的原来的矩阵$B$和$A$。基于这个假设，可以得到下面关于两个矩阵的梯度值，

![image-20250213193434705](https://gitee.com/fbanhua/figurebed/raw/master/images/20250213193434749.png)

另外可以得到梯度更新前后的模型输出变化为$\Delta f_{t}$，这里标出了三个项：第一个项表示固定a或固定b时**对模型输出的产生的变化关于学习率是线性的**，而两个矩阵的参数同时参与模型更新时，**学习率对模型输出产生的变化则是平方影响的**。如果$\Delta f_{t}=\Theta(1)$，即模型变化与模型宽度无关时，则公式的三个项中至少有一个项是$\Theta(1)$的。

在微调的理想情况下，我们希望第一和第二个项都是$\Theta(1)$的，否则两个矩阵中就会**有一个没有被有效更新**（相当于固定了一个矩阵，只对另一个矩阵进行的训练），也就是当第一和第二个项都是$\Theta(1)$时，两个矩阵都对模型的更新起了效果（两个矩阵都有效参与了特征学习），这在论文中被称为是“**LoRA是高效的**”。

在论文的一通数学证明中，作者给出了这篇论文的第一个命题：

![image-20250213195949559](https://gitee.com/fbanhua/figurebed/raw/master/images/20250213195949622.png)

这个命题的含义是：**当两个矩阵的参数按照上述提到的两个初始化方法初始化，并且学习率和模型宽度的某个次方相关时，$\Delta f_{t}$中的前两个项就无法保持$\Theta(1)$，也就是这时的“LoRA是不够高效的”**。基于这个命题，作者认为在原始的LoRA中缺少了一些关键的参数设置！！

![image-20250213200958798](https://gitee.com/fbanhua/figurebed/raw/master/images/20250213201415390.png)

又是一通数学证明，论文认为**为不同的矩阵分配不同的学习率**时可以使$\Delta f_{t}$中的所有项都是$\Theta(1)$的，这就得出了这篇论文的第二个命题：**只要两个矩阵的学习率分别符合$\eta_{a}=\Theta(n^{-1})$和$\eta_{b}=\Theta(1)$时（b的学习率比a的学习率大得多），就可以使微调过程中两个矩阵都学习到有效特征。**

在论文后面的部分中，可以简单理解为对上述的过程进行更加深入的分析，但是结论还是不变的。由于本文主要是对原理进行简单的介绍，所有就重复一下论文中的结论：**对两个矩阵使用同样的学习率是无法学到有效特征的，按照前面的推导设置不同的学习率比例才能使两个矩阵同时学到有效特征。**

## 参数

```yaml
finetuning_type: lora  # lora微调
lora_target: all
lora_rank: 8  # 16
loraplus_lr_ratio: 10
```

## 实验

这里与前面的原始LoRA对比可以看出，在相同的训练轮数中，**训练集上会有非常明显的损失值二次下降的过程，说明确实是比原始LoRA学习到了更多的特征。**

![image-20250213180002359](https://gitee.com/fbanhua/figurebed/raw/master/images/20250213180002416.png)

![image-20250213180124950](https://gitee.com/fbanhua/figurebed/raw/master/images/20250213180125003.png)

![image-20250214102850143](https://gitee.com/fbanhua/figurebed/raw/master/images/20250214102850212.png)

# rsLoRA

## 原理

**论文链接：**[A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA](https://arxiv.org/abs/2312.03732)

在LoRA微调中，实际上两个矩阵相乘后前面还要乘以一个缩放因子，即完整的公式为：
$$
W+\gamma_rBA,
$$
其中$\gamma_r$可以看做是关于所设置的`lora_rank`的一个函数，我们需要保证$\gamma_r$的设置是符合当前的训练的。为此，论文首先提出了一个关于**秩稳定**（rank-stabilized）的定义，即**前向传播稳定性和反向传播稳定性**（下面提到的适配器即为微调过程中的神经网络模块，也就是LoRA中的两个低秩矩阵）：

- **前向传播稳定性：**如果适配器输入中的每个元素是独立同分布的，并且每个元素的m阶矩都是$\Theta_{r}(1)$，那么适配器输出的每个元素的m阶矩也保持为$\Theta_{r}(1)$
- **反向传播稳定性：**如果损失函数对适配器输出的梯度在每个元素上为$\Theta_{r}(1)$，则损失函数对适配器输入的梯度在每个元素上也保持为$\Theta_{r}(1)$

![image-20250213204413423](https://gitee.com/fbanhua/figurebed/raw/master/images/20250213204413483.png)

基于这个定义，论文证明了一个定理：在使用[前面](#LoRA+)提到的第一种初始化方法时，**当且仅当$\gamma_r$的收敛速率属于$\Theta_r(\frac1{\sqrt{r}})$时，所有适配器是秩稳定的。**

![image-20250213205056153](https://gitee.com/fbanhua/figurebed/raw/master/images/20250213205056223.png)

## 参数

```yaml
finetuning_type: lora  # lora微调
lora_target: all
use_rslora: true
lora_rank: 8  # 16
flash_attn: fa2
```

## 实验

![image-20250213233414608](https://gitee.com/fbanhua/figurebed/raw/master/images/20250213233414689.png)

![image-20250214102403021](https://gitee.com/fbanhua/figurebed/raw/master/images/20250214102403121.png)

![image-20250214103130680](https://gitee.com/fbanhua/figurebed/raw/master/images/20250214103130752.png)



# DoRA

## 原理

**论文链接：**[DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)

如果说把[LoRA+](#LoRA+)看做是把一起更新的两个矩阵分开来进行看待，那么DoRA就是把权重矩阵更新时的**更新方向和更新大小**分开来讨论。在LoRA中，微调过程需要同时关注更新大小（量级）和方向两个部分；而在DoRA中，微调过程强制**参数专注于方向方面的学习**，而量级将作为独立的可调参数。

<img src="https://gitee.com/fbanhua/figurebed/raw/master/images/20250214110002340.png" alt="image-20250214105307382" style="zoom: 80%;" />

对照原论文中的图如上，首先原始权重可以进行按照公式进行分解得到**量级和方向**：
$$
W=m\frac{V}{||V||_c}=\|W\|_c\frac{W}{||W||_c},
$$
也就是对预训练权重进行分解，得到原权重矩阵的**量级参数**和**方向参数**，其中量级参数将初始化为可训练参数，方向参数则使用LoRA方法进行更新：
$$
W'=\underline{m}\frac{V+\Delta V}{||V+\Delta V||_c}=\underline{m}\frac{W_0+\underline{BA}}{||W_0+\underline{BA}||_c},
$$
其中**带下划线的部分就是可训练参数**。按照这个流程进行一次训练并合并所有参数后就可以按照LoRA的方法更新一次参数。

## 参数

```yaml
finetuning_type: lora  # lora微调
lora_target: all
lora_rank: 8
use_dora: true
```

## 实验



# PiSSA

## 原理

**论文链接：**[PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models](https://arxiv.org/abs/2404.02948)

[LoRA+](#LoRA+)中讨论过一次关于两个矩阵参数初始化的设置，但是重点放在了两个矩阵学习率的设置上，而PiSSA就是对两个矩阵的参数初始化方法进行了讨论。

## 参数

```yaml
finetuning_type: lora  # lora微调
lora_target: all
lora_rank: 8
pissa_init: true
```

## 实验



# 结果对比

|  微调方法   | 峰值显存(MB) | 平均显存(MB) | 最佳Step | Best Eval Loss | Final Train Loss | 训练时间 | 评测结果（Best） | 评测结果（Final） |
| :---------: | :----------: | :----------: | :------: | :------------: | :--------------: | :------: | :--------------: | :---------------: |
|   LoRA(8)   |    24,068    |  23,577.88   |   3200   |     0.3769     |      0.3825      | 01:00:22 |                  |                   |
|  LoRA+(8)   |    23,974    |  23,709.56   |   3200   |     0.3839     |      0.3116      | 01:49:26 |                  |                   |
|  rsLoRA(8)  |    24,084    |  23,742.89   |   3200   |     0.3761     |      0.3506      | 01:44:04 |                  |                   |
|  LoRA(16)   |    24,097    |  23,435.58   |   3200   |     0.3752     |      0.3582      | 00:52:54 |                  |                   |
|  LoRA+(16)  |    24,114    |  23,221.32   |   3000   |     0.3695     |      0.2949      | 01:16:07 |                  |                   |
| rsLoRA(16)  |    24,073    |  23,720.90   |   3200   |     0.3809     |      0.3038      | 01:35:28 |                  |                   |
|  LoRA(512)  |    24,040    |  23,804.91   |   200    |     0.4358     |      0.2852      | 01:16:31 |                  |                   |
| LoRA+(512)  |    24,014    |  23,495.57   |   6742   |     0.6344     |      0.5576      | 01:16:03 |                  |                   |
| rsLoRA(512) |    24,035    |  23,684.49   |   200    |      1.34      |      2.8987      | 01:17:45 |                  |                   |

