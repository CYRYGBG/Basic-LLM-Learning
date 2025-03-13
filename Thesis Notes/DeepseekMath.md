## 简介

**论文链接：**[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)

之前在[文章](https://zhuanlan.zhihu.com/p/29046223663)中学习了强化学习在LLM中的应用和简单的GRPO复现，并且实验发现确实是有一定效果，所以决定直接读一遍DeepseekMath的论文，也就是提出了GRPO算法的论文，同时写下了这篇笔记。

这里直接贴上读完论文的感悟：第一点，**数据集的质量和数量永远是重中之重！**论文中仅使用所收集到的高质量的大规模数据集训练的模型就已经取得了很好的效果；第二点，**算法的设计十分重要！**论文中提出的GRPO算法的思想简单但使用，直接用一个“group”生成的方法就简化了广义优势估计函数的计算和增大了模型的样本学习量。

## Math Pre-Training

### 数据收集

DeepseekMath中最核心也是最关键的一点就是采集了**高质量的数学数据集**，论文中对这个数据采集流程的描述和图示都是**以一个小型的高质量数据集作为“Seed”**，然后构建一个**可迭代的算法**使这个种子生根发芽成长为一棵大树，也就是最后采集到的更大的高质量数据集。

![数据采集流程](https://gitee.com/fbanhua/figurebed/raw/master/images/20250313151201802.png)

**第一步，**论文中以[OpenWebMath](https://github.com/keirp/OpenWebMath)作为初始的种子数据集，并从中**抽取500000个样本**作为正样本，再从另一个数据集[Common Crawl](https://commoncrawl.org/)同样抽取500000个网页作为负样本构成训练开源模型[FastText](https://fasttext.cc/)作为一个**快速的分类器**，用于判别某个网页是否为**数学相关**的网页。

![FastText训练集的构建](https://gitee.com/fbanhua/figurebed/raw/master/images/20250313151735468.png)

**第二步，**论文对经过**去重操作**的数据集[Common Crawl](https://commoncrawl.org/)（还剩40B网页）中所有的网页使用训练好的FastText模型进行预测，然后根据预测分数的高低进行排序。在第一轮迭代中，标记**前40B tokens**的内容为数学相关的网页。

![数据筛选](https://gitee.com/fbanhua/figurebed/raw/master/images/20250313152651964.png)

**第三步第四步，**由于前面的过程存在一个问题：训练FastText模型的数据集没有充分的多样性，所以还会有很多网页是数学相关的但是没有被分类正确。为了解决这个问题，论文找出了第一轮迭代中有**超过10%**的网页被标记为数学相关的网址（URL），然后把这些网址下的所有网页通通标记为数学相关的网页，然后**手动把这些网页添加到种子数据集**中。这个方法可以扩充原始的种子数据集，最后种子数据集已经“成长”为一个大型的数学网页数据集：有35.5 Million的数学相关网页，一共120B tokens的数据集。

另外，这个迭代算法在**迭代了4次之后就结束了**，因为在第4轮迭代之后，发现有98%的网页已经被收集为数学相关的网页了。

![迭代结果](https://gitee.com/fbanhua/figurebed/raw/master/images/20250313153401458.png)

同时，为了避免所收集的数据集含有一些LLM评估中测试集的数据，论文提到使用了一些匹配算法来对所有的测试集进行匹配，去除了满足匹配条件的网页。

### 质量评估

为了评估所收集到的数据集的质量，论文使用了Deepseek结构的1.3B模型在MathPile、OpenWebMath、Proof-Pile-2和DeepseekMath Corpus四个数据集上进行了**预训练**，然后对预训练后的模型进行评估，得出了结论：**DeepseekMath Corpus数据集的数据量是最多的，预训练后的模型效果也是最好的**。

![预训练模型评估结果](https://gitee.com/fbanhua/figurebed/raw/master/images/20250313154155836.png)

论文认为，数据集的高质量来源于**语言的多样性**（相比于原来的OpenWebMath数据集只有英文，新收集的数据集中即有英文又有中文，还有一小部分其他的语言）和**数据集的规模**（这个显然，比别的数据集大很多）。

### DeepSeekMath-Base

论文基于DeepSeek-Coder-Base-v1.5 7B模型对DeepSeekMath-Base 7B模型进行参数初始化，然后根据前面所采集到的DeepseekMath Corpus数据集上进行预训练。预训练数据集的**构成**为：56%的DeepseekMath Corpus，4%的[AlgebraicStack](https://huggingface.co/datasets/typeof/algebraic-stack)，10%的arXiv，20%的Github代码和10%的自然语言数据集。

![预训练数据集](https://gitee.com/fbanhua/figurebed/raw/master/images/20250313154912825.png)

这一部分后面的都是关于DeepSeekMath-Base的评估结果分析，无外乎以下几点：

- 使用chain-of-thought prompting的**数学推理能力**比别的模型牛
- 使用few-shot program-of-thought prompting的**用工具解答数学题的能力**也比别的模型牛
- **形式化证明和自然语言理解的能力**和别的模型不相上下

## DeepSeekMath-Instruct

在完成数据收集、预训练后，下一步就是指令监督微调了，这里论文使用了**基于不同方法解决题目**的数据集来进行微调。这里的“不同方法”就包括：**chain-of-thought**，**program-of-thought**和**tool-integrated reasoning format**。

![微调数据集](C:\Users\cyr\AppData\Roaming\Typora\typora-user-images\image-20250313160623720.png)

## Reinforcement Learning

关于强化学习的内容我之前有写[一篇文章](https://zhuanlan.zhihu.com/p/29046223663)专门介绍过，这里就不再赘述了，只做简单必要的介绍。

![GRPO](https://gitee.com/fbanhua/figurebed/raw/master/images/20250313162524862.png)

**GRPO中没有使用价值网络来对输出进行评估**（PPO一般用在actor-critic上，需要有一个价值网络来对actor的行为进行评判，value model和reward model的区别在于，reward model只是按照当前的输出来给出当前这一步的奖励，相当于来自环境的反馈，而value model需要判断这个输出**是否超过了模型应该有的平均表现**，所以前面提到，叫**优势估计**），而是当LLM模型（策略网络）输入一个$q$后，推理得到多个输出$o_1, o_2,...o_G$，然后通过reward model对每个输出计算当前输出的奖励，再根据这个奖励**计算优势估计结果**。

## 一些发现

论文在最后分享并通过实验证明了他们的一些发现，可以概括为以下关键的几点：

- 对模型的**代码训练**可以提高模型的数学推理能力
- 使用**arXiv论文**进行训练对模型的提高有限甚至是有害
- 强化学习中使用**online sampling**比offline sampling效果要好（online sampling指使用最新的策略模型生成的结果进行训练；offline sampling指仅使用SFT模型生成的结果进行训练）





**PS：**到这里这篇论文的基本内容就读完了，对笔者自己来说最大的感悟还是数据集方面的，因为现在在做一个项目，自己琢磨的算法在别的数据集上都还能有比较好的效果，就是在学校自己的一个数据集上效果极差，自信心受挫十分严重。后面直接用了十来个模型，包括24年Neurips的一个该领域的预训练大模型之后才发现，没有一个模型能在这个数据集上效果好的，自信心才恢复一点，数据集的质量和本身的可分性（针对分类数据集）还是十分重要啊！