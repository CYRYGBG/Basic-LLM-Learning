<<<<<<< HEAD
TRPO前的部分为学习[【王树森】深度强化学习(DRL)](https://www.bilibili.com/video/BV12o4y197US?spm_id_from=333.788.videopod.episodes&vd_source=cbcc0f860e979710ca238ab1cdbe2a75)的简要笔记[课程github](https://github.com/wangshusen/DRL)，从PPO开始的部分则是学习原论文的笔记。学习过程中的一些基础代码则是基于[Huggingface Deep RL Course](https://huggingface.co/learn/deep-rl-course/unit0/introduction)（本文不涉及关于强化学习具体代码的实现）。

## 参考链接

[简要的DPO、IPO、KTO实验效果对比:  Preference Tuning LLMs with Direct Preference Optimization Methods](https://huggingface.co/blog/pref-tuning)

[对RLHF的简要介绍：Illustrating Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf)

[数据集格式](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/data_preparation.html#id10)

[综述：A Comprehensive Survey of LLM Alignment Techniques: RLHF, RLAIF, PPO, DPO and More](https://arxiv.org/abs/2407.16216)

[深度对比: SFT、ReFT、RLHF、RLAIF、DPO、PPO](https://www.cnblogs.com/Microsoftdeveloper/articles/18635866)

[油管视频：[GRPO Explained] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://www.youtube.com/watch?v=bAWV_yrqx4w)

[GRPO From Scratch](https://github.com/aburkov/theLMbook/blob/main/GRPO_From_Scratch_Multi_GPU_DataParallel_Qwen_2_5_1_5B_Instruct.ipynb)

## 强化学习简要基础

### 符号解释

![符号总结](https://gitee.com/fbanhua/figurebed/raw/master/images/20250303102524283.png)

- **动作价值函数：**对**当前状态$s_t$**下，**进行一个动作$a_t$后**所获得的未来return的期望值
- **状态价值函数：**当前状态下，**对所有动作的未来return计算期望值**，表示当前状态的好坏，也可以与其他策略比较得出不同策略在当前状态下的差别
- **最优动作价值函数：**（图中没有给出），公式是$Q^\star(s_t,a_t)=\max_\pi Q_\pi(s_t,a_t)$，表示对**当前状态$s_t$**下，**进行一个动作$a_t$后**，在所有策略中可能得到的动作价值的最高值，也就是这个动作**在所有策略中可以带来的最高的平均收益**

强化学习的基础假设有：

- 随机性来源于两个地方：策略函数和状态转移函数
- 模型学习的依据：做出决策后环境的反馈（return）
- 评价决策的好坏：动作价值函数
- 评价状态的好坏：状态价值函数

### 价值学习

![TD算法流程](https://gitee.com/fbanhua/figurebed/raw/master/images/20250303102524791.png)

> 价值学习的核心是通过一个神经网络去逼近最优动作价值函数。

TD算法的核心是：模型**一开始**可以给出一个**最终的分数预测**，在每一步中实施动作后，实施该动作的分数就已经确定了（相当于原来只是预测，但是某个动作实施后分数就是可观测到的了）。所以继续**从“当前状态”预测**最终的分数时，**由于一部分分数已经是确定了不可改变了，这个预测肯定与最开始的预测有所不同**，也就产生了误差，根据这个误差构建的损失函数就可以对模型的参数进行更新了。这一步可以在每一次模型做出决策后进行。

### 策略学习

![策略学习](https://gitee.com/fbanhua/figurebed/raw/master/images/20250303102523153.png)

策略学习的核心是**使用神经网络近似策略函数$\pi$**（$\theta$即表示神经网络中的参数），目标函数就是在使得在**当前策略函数下状态价值函数最大化**（所以这里的不是梯度下降，是梯度上升），这里的梯度计算包含两种形式，下面是关于当前状态$s_t$的状态价值函数表示：
$$
V_\pi(s_t)=\mathbb{E}_A[Q_\pi(s_t,A)],
$$
如果是**离散**的动作行为，这里的计算就是对每个动作进行累加求和；如果是**连续**的动作行为，则是进行积分计算。在离散的行为中，对应的梯度公式为：
$$
\frac{\partial V(s;\mathbf{\theta})}{\partial\mathbf{\theta}}=\sum_a\frac{\partial\pi(a|s;\mathbf{\theta})}{\partial\mathbf{\theta}}\cdot Q_\pi(s,a),
$$
在连续的行为中，则表示为：
$$
\frac{\partial V(s;\mathbf{\theta})}{\partial\mathbf{\theta}}=\mathbb{E}_{A\sim\pi(\cdot|s;\mathbf{\theta})}\left[\frac{\partial\log\pi(A|s,\mathbf{\theta})}{\partial\mathbf{\theta}}\cdot Q_\pi(s,A)\right],
$$
由于这里的期望需要对所有连续的行为进行求积分，实际计算中比较困难，所以一般是**使用蒙特卡洛方法**来近似（在得到完整的行为决策后，最后再对每一步进行计算）。

![算法流程](https://gitee.com/fbanhua/figurebed/raw/master/images/20250303102523450.png)

### Actor-Critic Method

![算法流程](https://gitee.com/fbanhua/figurebed/raw/master/images/20250303102523865.png)

在这个方法中，**策略函数和动作价值函数都是用神经网络近似的**，整个算法流程更详细的讲就是：

1. 使用策略函数根据当前状态对动作进行采样
2. 实施动作后获取新的状态和来自环境的奖励
3. 根据新的状态采样得到下一步动作，但是不真的实施
4. 由于已经实施了一步实际的动作和获取了来自环境的真实return，所以可以使用价值学习中的TD算法对**动作价值函数**的参数进行更新（梯度下降）
5. 使用对当前已实施动作的动作价值函数的评分对**策略函数**的参数进行更新（梯度上升）

在整个训练过程中，模型的**提升**来源于动作价值函数对环境return的拟合，而**训练结束后获取**的则是策略函数。

### TRPO(信任区域策略优化)

原文链接：[Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)

![置信域算法](https://gitee.com/fbanhua/figurebed/raw/master/images/20250303102523352.png)

其中$L(\mathbf{\theta}\mid\mathbf{\theta}_{\mathrm{old}})$表示对目标函数的近似，而
$$
\mathcal{N}(\mathbf{\theta}_{\mathrm{old}})=\left\{\mathbf{\theta}\left||\mathbf{\theta}-\mathbf{\theta}_{\mathrm{old}}\right||_2\leq\Delta\right\},
$$
表示更新后的参数应该落在的区间。

简要的说，就是有时需要进行梯度下降或者梯度上升的目标函数的梯度不好求，所以**就用一个近似函数在当前的$\theta_{old}$来对目标函数进行逼近后进行梯度计算**，然后因为函数的逼近在某个范围内才是误差最小的（比如对某个函数在零点处进行Taylor展开），所以就会存在一个置信域（Trust Region），需要**保证更新后的参数还落在置信域中**才是误差最小的（或者说是误差可接受的）。

![TRPO算法](https://gitee.com/fbanhua/figurebed/raw/master/images/20250303102525092.png)

在置信域算法的基础上，可以对策略函数的目标函数做近似，然后再在参数的置信域中进行梯度更新。而计算两个参数距离的方法在视频中提到有两种：**直接计算范数和计算KL散度**（计算两个概率分布的距离）。同时由于需要对动作价值函数做近似，所以策略函数的参数在**决策过程中不会进行更新**，只有当某一局完全结束后，计算得到动作价值函数的采样之后才能计算出近似函数进行更新。

**这里存在一个疑问：**，在TRPO和后续介绍的PPO算法中，都提到了需要通过某种方法计算旧模型参数和新模型参数之间的**“距离”**（也许是输出的概率分布之间的差异，也有可能直接计算两个模型的范数），**那就会产生一个问题**：在进行梯度更新之前，怎么得到新的模型参数和输出的概率分布？如果没有新的模型参数和输出的概率分布，又怎么计算出损失函数？

这个问题在论文的**附录C部分**有明确的数学过程表示，**基本方法就是通过使用逼近的方法先搜索出一个梯度更新方向，然后再在这个方向上使用line search，在保证满足非线性约束的同时优化目标函数。**

### PPO(近端策略优化)

**原文链接：**[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

在TRPO的思想中，就是通过设定置信域的方法对参数的更新范围进行限制，防止模型模型参数的更新幅度过大导致训练不稳定，**PPO则是更进一步用了更好的限制**。首先来看一下原本的目标函数：
$$
L^{PG}(\theta)=\hat{\mathbb{E}}_t\left[\log\pi_\theta(a_t\mid s_t)\hat{A}_t\right]
$$
对上述的目标函数计算近似梯度，计算公式为：
$$
\hat{g}=\hat{\mathbb{E}}_t\left[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\hat{A}_t\right],
$$
（[附上公式推导链接](https://huggingface.co/learn/deep-rl-course/unit4/pg-theorem)），其中$\pi_\theta$即为参数为$\theta$的策略函数，$\hat{A}_t$表示**当前时间步$t$下**的**某个动作相对于平均表现的优势**。而**在TRPO中**，对应的**目标函数**可以简化为
$$
\underset{\theta}{\operatorname*{\mathrm{maximize}}}\quad\hat{\mathbb{E}}_t\left[\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}\hat{A}_t\right],

\\
\text { subject to } \quad \hat{\mathbb{E}}_{t}\left[\mathrm{KL}\left[\pi_{\theta_{\text {old }}}\left(\cdot \mid s_{t}\right), \pi_{\theta}\left(\cdot \mid s_{t}\right)\right]\right] \leq \delta,
$$
可以看出，这是参数更新受限的目标函数，并且新的参数和旧的参数对某个动作分配的概率差距越大（假设是分配给更优动作的概率，所以新参数的概率应该更大），目标函数的值也会越高。

对应的，在PPO的论文中提出的目标函数则是
$$
L^{C L I P}(\theta)=\hat{\mathbb{E}}_{t}\left[\min \left(r_{t}(\theta) \hat{A}_{t}, \operatorname{clip}\left(r_{t}(\theta), 1-\epsilon, 1+\epsilon\right) \hat{A}_{t}\right)\right],
$$
其中$r_{t}(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}$，也就是$\min$左边的第一项和TRPO中的目标函数表示相同；$\operatorname{clip}$函数表示将$r_{t}(\theta)$限制在指定的区间内，可以参考论文中的图进行理解：

- 如果$r_{t}(\theta)$的值在区间$[1-\epsilon, 1+\epsilon]$中，那么就保留原来的$r_{t}(\theta)$，也就是$\min$函数两边的值一样
- 如果$r_{t}(\theta)$的值小于$1-\epsilon$，那么就直接使用$1-\epsilon$作为系数；大于则使用$1+\epsilon$作为系数



![CLIP](https://gitee.com/fbanhua/figurebed/raw/master/images/20250303151021696.png)

另外，在论文中还提出了**另一种实现**，首先根据TRPO中的目标函数和对应的约束条件，可以把目标函数转换为
$$
\operatorname*{maximize}_{\theta}\hat{\mathbb{E}}_{t}\left[\frac{\pi_{\theta}(a_{t}\mid s_{t})}{\pi_{\theta_{\mathrm{old}}}(a_{t}\mid s_{t})}\hat{A}_{t}-\beta\operatorname{KL}[\pi_{\theta_{\mathrm{old}}}(\cdot\mid s_{t}),\pi_{\theta}(\cdot\mid s_{t})]\right],
$$
也就是**把约束条件加入到了目标函数中**并使用$\beta$作为超参数，然后设计了如下图所示的自适应更新算法

![image-20250303152900525](https://gitee.com/fbanhua/figurebed/raw/master/images/20250303152900602.png)

其中$d_{targ}$即为类似前一种实现中$\epsilon$的超参数。

总的来说，PPO方法就是防止模型对分数的追求过于激进，需要对它的参数更新和学习加入一定的限制。

## 关于LLM的强化学习

### DPO(直接偏好优化)

**原文链接：**[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)

DPO是专门针对LLM中人类偏好对齐的算法，是强化学习在LLM中的发展。PPO中约束条件简化进目标的函数后，目标函数基本可以分为两部分：**期望奖励部分**和**更新幅度惩罚部分**。论文中提到，该目标函数调优后得到的最终策略（即为LLM模型）可以表示为：
$$
\pi_r(y\mid x)=\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y\mid x)\exp\left(\frac{1}{\beta}r(x,y)\right),
$$
其中**$\pi_{\mathrm{ref}}(y\mid x)$**，即为经过SFT训练后的模型，已经具有一定的指令回答能力，$x$表示prompts，$y$表示模型的输出，在LLM的强化学习中，把整个**LLM模型视作策略函数**，输入的**prompts视为状态**，最终目标是根据设定好的奖励，希望模型在当前状态（prompts）下**输出更符合人类偏好的结果**（RL中“更好的决策”）；**$\beta$**是关于惩罚项的超参数；$r(x,y)$表示在当前输入下对模型输出的评分，即**奖励函数**；另外，
$$
Z(x)=\sum_y\pi_\mathrm{ref}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right),
$$
通过这个策略函数反推，可以得到奖励函数的表示：
$$
r(x,y)=\beta\log\frac{\pi_{r}(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+\beta\log Z(x),
$$
假如模型生成了两个句子$y_1$和$y_2$，论文中用Bradley-Terry模型来对偏好进行建模，公式为
$$
p^*(y_1\succ y_2\mid x)=\frac{\exp\left(r^*(x,y_1)\right)}{\exp\left(r^*(x,y_1)\right)+\exp\left(r^*(x,y_2)\right)},
$$
这个公式就表示了人类偏好的分布，把前面反推得到的奖励函数代入这个公式，可以得到
$$
p^*(y_1\succ y_2\mid x)=\frac{1}{1+\exp\left(\beta\log\frac{\pi^*(y_2|x)}{\pi_{\mathrm{ref}}(y_2|x)}-\beta\log\frac{\pi^*(y_1|x)}{\pi_{\mathrm{ref}}(y_1|x)}\right)},
$$
既然任务的目标是希望模型能够更大概率的输出人类喜欢的回答，所以可以表示为概率的形式，那么就可以**使用最大似然估计来使这个概率最大化**，对应的论文推出了下面的公式：
$$
\mathcal{L}_{\mathrm{DPO}}(\pi_\theta;\pi_{\mathrm{ref}})=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right)\right],
$$
这也就是**DPO中的目标函数**，在模型的训练中，梯度下降只需要根据这个目标函数进行就可以了，直接避免了强化学习中奖励建模和策略优化等多阶段的优化过程，把模型训练**简化到只有按照目标函数来进行梯度下降一个阶段（监督学习）**。

### GRPO(Group Relative Policy Optimization)

**原文链接：**[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)

论文首先回顾了一下PPO的公式：
$$
\mathcal{J}_{P P O}(\theta)=\mathbb{E}\left[q \sim P(Q), o \sim \pi_{\theta_{o l d}}(O \mid q)\right] \frac{1}{|o|} \sum_{t=1}^{|o|} \min \left[\frac{\pi_{\theta}\left(o_{t} \mid q, o_{<t}\right)}{\pi_{\theta_{o l d}}\left(o_{t} \mid q, o_{<t}\right)} A_{t}, \operatorname{clip}\left(\frac{\pi_{\theta}\left(o_{t} \mid q, o_{<t}\right)}{\pi_{\theta_{o l d}}\left(o_{t} \mid q, o_{<t}\right)}, 1-\varepsilon, 1+\varepsilon\right) A_{t}\right],
$$
与前面PPO中公式的差别就是这里随机采样了很多个输出来求平均进行计算，$A_t$是广义优势估计（Generalized Advantage Estimation，**GAE**）。随后论文贴出了两个方法的对比：
![image-20250304150819001](https://gitee.com/fbanhua/figurebed/raw/master/images/20250304150819141.png)

可以看到，**GRPO中直接没有使用价值网络来对输出进行评估**（PPO一般用在actor-critic上，需要有一个价值网络来对actor的行为进行评判，value model和reward model的区别在于，reward model只是按照当前的输出来给出当前这一步的奖励，相当于来自环境的反馈，而value model需要判断这个输出**是否超过了模型应该有的平均表现**，所以前面提到，叫**优势估计**），而是当LLM模型（策略网络）输入一个$q$后，推理得到多个输出$o_1, o_2,...o_G$，然后通过reward model对每个输出计算当前输出的奖励，GRPO直接通过下面的公式代替广义估计函数：
$$
\hat{A}_{i, t}=\widetilde{r}_{i}=\frac{r_{i}-\operatorname{mean}(\mathbf{r})}{\operatorname{std}(\mathbf{r})},
$$
这样就不用构建一个巨大的value model。对应的，**GRPO认为经过监督微调后的SFT模型$\pi_{ref}$中已经具有一定的知识**，所以强化学习只是为了从中把知识“释放”出来，所以在下面目标函数的最后**增加了与SFT模型$\pi_{ref}$距离的惩罚项。**
$$
\begin{aligned}
\mathcal{J}_{GRPO}(\theta) & =\operatorname{E}[q\sim P(Q),\{o_{i}\}_{i=1}^{G}\sim\pi_{\theta_{old}}(O|q)] \\
 & \frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_{i}|}\sum_{t=1}^{|o_{i}|}\left\{\min\left[\frac{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q,o_{i,<t})}\hat{A}_{i,t},\mathrm{clip}\left(\frac{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q,o_{i,<t})},1-\varepsilon,1+\varepsilon\right)\hat{A}_{i,t}\right]-\beta\mathbf{D}_{KL}\left[\pi_{\theta}||\pi_{ref}\right]\right\}
\end{aligned}
$$

### RLHF(基于人类反馈的强化学习)

这一部分主要是将强化学习怎么在LLM中发挥作用的。

**提出论文**：[Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325)

**更多应用**：[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

![image-20250304154128804](https://gitee.com/fbanhua/figurebed/raw/master/images/20250304154128963.png)

上图即为PPO用于训练模型输出更符合人类喜好的结果，整个过程分三步：

- 模型输出成对的结果，人工选取最佳结果
- 从人类反馈中学习奖励模型，用于判断哪个输出的结果更好
- 针对奖励模型优化策略函数（LLM）

![image-20250304154914254](https://gitee.com/fbanhua/figurebed/raw/master/images/20250304154914422.png)

在引用的第二篇论文中，训练步骤为

- 人类编写一些回答来对模型进行监督微调
- 模型输出多个结果，人类对模型输出结果的好坏进行排序，并基于该结果训练奖励模型
- 使用奖励模型训练LLM
=======
TRPO前的部分为学习[【王树森】深度强化学习(DRL)](https://www.bilibili.com/video/BV12o4y197US?spm_id_from=333.788.videopod.episodes&vd_source=cbcc0f860e979710ca238ab1cdbe2a75)的简要笔记[课程github](https://github.com/wangshusen/DRL)，从PPO开始的部分则是学习原论文的笔记。学习过程中的一些基础代码则是基于[Huggingface Deep RL Course](https://huggingface.co/learn/deep-rl-course/unit0/introduction)（本文不涉及关于强化学习具体代码的实现）。

## 参考链接

[简要的DPO、IPO、KTO实验效果对比:  Preference Tuning LLMs with Direct Preference Optimization Methods](https://huggingface.co/blog/pref-tuning)

[对RLHF的简要介绍：Illustrating Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf)

[数据集格式](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/data_preparation.html#id10)

[综述：A Comprehensive Survey of LLM Alignment Techniques: RLHF, RLAIF, PPO, DPO and More](https://arxiv.org/abs/2407.16216)

[深度对比: SFT、ReFT、RLHF、RLAIF、DPO、PPO](https://www.cnblogs.com/Microsoftdeveloper/articles/18635866)

[油管视频：[GRPO Explained] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://www.youtube.com/watch?v=bAWV_yrqx4w)

[GRPO From Scratch](https://github.com/aburkov/theLMbook/blob/main/GRPO_From_Scratch_Multi_GPU_DataParallel_Qwen_2_5_1_5B_Instruct.ipynb)

## 强化学习简要基础

### 符号解释

![符号总结](https://gitee.com/fbanhua/figurebed/raw/master/images/20250303102524283.png)

- **动作价值函数：**对**当前状态$s_t$**下，**进行一个动作$a_t$后**所获得的未来return的期望值
- **状态价值函数：**当前状态下，**对所有动作的未来return计算期望值**，表示当前状态的好坏，也可以与其他策略比较得出不同策略在当前状态下的差别
- **最优动作价值函数：**（图中没有给出），公式是$Q^\star(s_t,a_t)=\max_\pi Q_\pi(s_t,a_t)$，表示对**当前状态$s_t$**下，**进行一个动作$a_t$后**，在所有策略中可能得到的动作价值的最高值，也就是这个动作**在所有策略中可以带来的最高的平均收益**

强化学习的基础假设有：

- 随机性来源于两个地方：策略函数和状态转移函数
- 模型学习的依据：做出决策后环境的反馈（return）
- 评价决策的好坏：动作价值函数
- 评价状态的好坏：状态价值函数

### 价值学习

![TD算法流程](https://gitee.com/fbanhua/figurebed/raw/master/images/20250303102524791.png)

> 价值学习的核心是通过一个神经网络去逼近最优动作价值函数。

TD算法的核心是：模型**一开始**可以给出一个**最终的分数预测**，在每一步中实施动作后，实施该动作的分数就已经确定了（相当于原来只是预测，但是某个动作实施后分数就是可观测到的了）。所以继续**从“当前状态”预测**最终的分数时，**由于一部分分数已经是确定了不可改变了，这个预测肯定与最开始的预测有所不同**，也就产生了误差，根据这个误差构建的损失函数就可以对模型的参数进行更新了。这一步可以在每一次模型做出决策后进行。

### 策略学习

![策略学习](https://gitee.com/fbanhua/figurebed/raw/master/images/20250303102523153.png)

策略学习的核心是**使用神经网络近似策略函数$\pi$**（$\theta$即表示神经网络中的参数），目标函数就是在使得在**当前策略函数下状态价值函数最大化**（所以这里的不是梯度下降，是梯度上升），这里的梯度计算包含两种形式，下面是关于当前状态$s_t$的状态价值函数表示：
$$
V_\pi(s_t)=\mathbb{E}_A[Q_\pi(s_t,A)],
$$
如果是**离散**的动作行为，这里的计算就是对每个动作进行累加求和；如果是**连续**的动作行为，则是进行积分计算。在离散的行为中，对应的梯度公式为：
$$
\frac{\partial V(s;\mathbf{\theta})}{\partial\mathbf{\theta}}=\sum_a\frac{\partial\pi(a|s;\mathbf{\theta})}{\partial\mathbf{\theta}}\cdot Q_\pi(s,a),
$$
在连续的行为中，则表示为：
$$
\frac{\partial V(s;\mathbf{\theta})}{\partial\mathbf{\theta}}=\mathbb{E}_{A\sim\pi(\cdot|s;\mathbf{\theta})}\left[\frac{\partial\log\pi(A|s,\mathbf{\theta})}{\partial\mathbf{\theta}}\cdot Q_\pi(s,A)\right],
$$
由于这里的期望需要对所有连续的行为进行求积分，实际计算中比较困难，所以一般是**使用蒙特卡洛方法**来近似（在得到完整的行为决策后，最后再对每一步进行计算）。

![算法流程](https://gitee.com/fbanhua/figurebed/raw/master/images/20250303102523450.png)

### Actor-Critic Method

![算法流程](https://gitee.com/fbanhua/figurebed/raw/master/images/20250303102523865.png)

在这个方法中，**策略函数和动作价值函数都是用神经网络近似的**，整个算法流程更详细的讲就是：

1. 使用策略函数根据当前状态对动作进行采样
2. 实施动作后获取新的状态和来自环境的奖励
3. 根据新的状态采样得到下一步动作，但是不真的实施
4. 由于已经实施了一步实际的动作和获取了来自环境的真实return，所以可以使用价值学习中的TD算法对**动作价值函数**的参数进行更新（梯度下降）
5. 使用对当前已实施动作的动作价值函数的评分对**策略函数**的参数进行更新（梯度上升）

在整个训练过程中，模型的**提升**来源于动作价值函数对环境return的拟合，而**训练结束后获取**的则是策略函数。

### TRPO(信任区域策略优化)

原文链接：[Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)

![置信域算法](https://gitee.com/fbanhua/figurebed/raw/master/images/20250303102523352.png)

其中$L(\mathbf{\theta}\mid\mathbf{\theta}_{\mathrm{old}})$表示对目标函数的近似，而
$$
\mathcal{N}(\mathbf{\theta}_{\mathrm{old}})=\left\{\mathbf{\theta}\left||\mathbf{\theta}-\mathbf{\theta}_{\mathrm{old}}\right||_2\leq\Delta\right\},
$$
表示更新后的参数应该落在的区间。

简要的说，就是有时需要进行梯度下降或者梯度上升的目标函数的梯度不好求，所以**就用一个近似函数在当前的$\theta_{old}$来对目标函数进行逼近后进行梯度计算**，然后因为函数的逼近在某个范围内才是误差最小的（比如对某个函数在零点处进行Taylor展开），所以就会存在一个置信域（Trust Region），需要**保证更新后的参数还落在置信域中**才是误差最小的（或者说是误差可接受的）。

![TRPO算法](https://gitee.com/fbanhua/figurebed/raw/master/images/20250303102525092.png)

在置信域算法的基础上，可以对策略函数的目标函数做近似，然后再在参数的置信域中进行梯度更新。而计算两个参数距离的方法在视频中提到有两种：**直接计算范数和计算KL散度**（计算两个概率分布的距离）。同时由于需要对动作价值函数做近似，所以策略函数的参数在**决策过程中不会进行更新**，只有当某一局完全结束后，计算得到动作价值函数的采样之后才能计算出近似函数进行更新。

**这里存在一个疑问：**，在TRPO和后续介绍的PPO算法中，都提到了需要通过某种方法计算旧模型参数和新模型参数之间的**“距离”**（也许是输出的概率分布之间的差异，也有可能直接计算两个模型的范数），**那就会产生一个问题**：在进行梯度更新之前，怎么得到新的模型参数和输出的概率分布？如果没有新的模型参数和输出的概率分布，又怎么计算出损失函数？

这个问题在论文的**附录C部分**有明确的数学过程表示，**基本方法就是通过使用逼近的方法先搜索出一个梯度更新方向，然后再在这个方向上使用line search，在保证满足非线性约束的同时优化目标函数。**

### PPO(近端策略优化)

**原文链接：**[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

在TRPO的思想中，就是通过设定置信域的方法对参数的更新范围进行限制，防止模型模型参数的更新幅度过大导致训练不稳定，**PPO则是更进一步用了更好的限制**。首先来看一下原本的目标函数：
$$
L^{PG}(\theta)=\hat{\mathbb{E}}_t\left[\log\pi_\theta(a_t\mid s_t)\hat{A}_t\right]
$$
对上述的目标函数计算近似梯度，计算公式为：
$$
\hat{g}=\hat{\mathbb{E}}_t\left[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\hat{A}_t\right],
$$
（[附上公式推导链接](https://huggingface.co/learn/deep-rl-course/unit4/pg-theorem)），其中$\pi_\theta$即为参数为$\theta$的策略函数，$\hat{A}_t$表示**当前时间步$t$下**的**某个动作相对于平均表现的优势**。而**在TRPO中**，对应的**目标函数**可以简化为
$$
\underset{\theta}{\operatorname*{\mathrm{maximize}}}\quad\hat{\mathbb{E}}_t\left[\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}\hat{A}_t\right],

\\
\text { subject to } \quad \hat{\mathbb{E}}_{t}\left[\mathrm{KL}\left[\pi_{\theta_{\text {old }}}\left(\cdot \mid s_{t}\right), \pi_{\theta}\left(\cdot \mid s_{t}\right)\right]\right] \leq \delta,
$$
可以看出，这是参数更新受限的目标函数，并且新的参数和旧的参数对某个动作分配的概率差距越大（假设是分配给更优动作的概率，所以新参数的概率应该更大），目标函数的值也会越高。

对应的，在PPO的论文中提出的目标函数则是
$$
L^{C L I P}(\theta)=\hat{\mathbb{E}}_{t}\left[\min \left(r_{t}(\theta) \hat{A}_{t}, \operatorname{clip}\left(r_{t}(\theta), 1-\epsilon, 1+\epsilon\right) \hat{A}_{t}\right)\right],
$$
其中$r_{t}(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}$，也就是$\min$左边的第一项和TRPO中的目标函数表示相同；$\operatorname{clip}$函数表示将$r_{t}(\theta)$限制在指定的区间内，可以参考论文中的图进行理解：

- 如果$r_{t}(\theta)$的值在区间$[1-\epsilon, 1+\epsilon]$中，那么就保留原来的$r_{t}(\theta)$，也就是$\min$函数两边的值一样
- 如果$r_{t}(\theta)$的值小于$1-\epsilon$，那么就直接使用$1-\epsilon$作为系数；大于则使用$1+\epsilon$作为系数



![CLIP](https://gitee.com/fbanhua/figurebed/raw/master/images/20250303151021696.png)

另外，在论文中还提出了**另一种实现**，首先根据TRPO中的目标函数和对应的约束条件，可以把目标函数转换为
$$
\operatorname*{maximize}_{\theta}\hat{\mathbb{E}}_{t}\left[\frac{\pi_{\theta}(a_{t}\mid s_{t})}{\pi_{\theta_{\mathrm{old}}}(a_{t}\mid s_{t})}\hat{A}_{t}-\beta\operatorname{KL}[\pi_{\theta_{\mathrm{old}}}(\cdot\mid s_{t}),\pi_{\theta}(\cdot\mid s_{t})]\right],
$$
也就是**把约束条件加入到了目标函数中**并使用$\beta$作为超参数，然后设计了如下图所示的自适应更新算法

![image-20250303152900525](https://gitee.com/fbanhua/figurebed/raw/master/images/20250303152900602.png)

其中$d_{targ}$即为类似前一种实现中$\epsilon$的超参数。

总的来说，PPO方法就是防止模型对分数的追求过于激进，需要对它的参数更新和学习加入一定的限制。

## 关于LLM的强化学习

### DPO(直接偏好优化)

**原文链接：**[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)

DPO是专门针对LLM中人类偏好对齐的算法，是强化学习在LLM中的发展。PPO中约束条件简化进目标的函数后，目标函数基本可以分为两部分：**期望奖励部分**和**更新幅度惩罚部分**。论文中提到，该目标函数调优后得到的最终策略（即为LLM模型）可以表示为：
$$
\pi_r(y\mid x)=\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y\mid x)\exp\left(\frac{1}{\beta}r(x,y)\right),
$$
其中**$\pi_{\mathrm{ref}}(y\mid x)$**，即为经过SFT训练后的模型，已经具有一定的指令回答能力，$x$表示prompts，$y$表示模型的输出，在LLM的强化学习中，把整个**LLM模型视作策略函数**，输入的**prompts视为状态**，最终目标是根据设定好的奖励，希望模型在当前状态（prompts）下**输出更符合人类偏好的结果**（RL中“更好的决策”）；**$\beta$**是关于惩罚项的超参数；$r(x,y)$表示在当前输入下对模型输出的评分，即**奖励函数**；另外，
$$
Z(x)=\sum_y\pi_\mathrm{ref}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right),
$$
通过这个策略函数反推，可以得到奖励函数的表示：
$$
r(x,y)=\beta\log\frac{\pi_{r}(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+\beta\log Z(x),
$$
假如模型生成了两个句子$y_1$和$y_2$，论文中用Bradley-Terry模型来对偏好进行建模，公式为
$$
p^*(y_1\succ y_2\mid x)=\frac{\exp\left(r^*(x,y_1)\right)}{\exp\left(r^*(x,y_1)\right)+\exp\left(r^*(x,y_2)\right)},
$$
这个公式就表示了人类偏好的分布，把前面反推得到的奖励函数代入这个公式，可以得到
$$
p^*(y_1\succ y_2\mid x)=\frac{1}{1+\exp\left(\beta\log\frac{\pi^*(y_2|x)}{\pi_{\mathrm{ref}}(y_2|x)}-\beta\log\frac{\pi^*(y_1|x)}{\pi_{\mathrm{ref}}(y_1|x)}\right)},
$$
既然任务的目标是希望模型能够更大概率的输出人类喜欢的回答，所以可以表示为概率的形式，那么就可以**使用最大似然估计来使这个概率最大化**，对应的论文推出了下面的公式：
$$
\mathcal{L}_{\mathrm{DPO}}(\pi_\theta;\pi_{\mathrm{ref}})=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right)\right],
$$
这也就是**DPO中的目标函数**，在模型的训练中，梯度下降只需要根据这个目标函数进行就可以了，直接避免了强化学习中奖励建模和策略优化等多阶段的优化过程，把模型训练**简化到只有按照目标函数来进行梯度下降一个阶段（监督学习）**。

### GRPO(Group Relative Policy Optimization)

**原文链接：**[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)

论文首先回顾了一下PPO的公式：
$$
\mathcal{J}_{P P O}(\theta)=\mathbb{E}\left[q \sim P(Q), o \sim \pi_{\theta_{o l d}}(O \mid q)\right] \frac{1}{|o|} \sum_{t=1}^{|o|} \min \left[\frac{\pi_{\theta}\left(o_{t} \mid q, o_{<t}\right)}{\pi_{\theta_{o l d}}\left(o_{t} \mid q, o_{<t}\right)} A_{t}, \operatorname{clip}\left(\frac{\pi_{\theta}\left(o_{t} \mid q, o_{<t}\right)}{\pi_{\theta_{o l d}}\left(o_{t} \mid q, o_{<t}\right)}, 1-\varepsilon, 1+\varepsilon\right) A_{t}\right],
$$
与前面PPO中公式的差别就是这里随机采样了很多个输出来求平均进行计算，$A_t$是广义优势估计（Generalized Advantage Estimation，**GAE**）。随后论文贴出了两个方法的对比：
![image-20250304150819001](https://gitee.com/fbanhua/figurebed/raw/master/images/20250304150819141.png)

可以看到，**GRPO中直接没有使用价值网络来对输出进行评估**（PPO一般用在actor-critic上，需要有一个价值网络来对actor的行为进行评判，value model和reward model的区别在于，reward model只是按照当前的输出来给出当前这一步的奖励，相当于来自环境的反馈，而value model需要判断这个输出**是否超过了模型应该有的平均表现**，所以前面提到，叫**优势估计**），而是当LLM模型（策略网络）输入一个$q$后，推理得到多个输出$o_1, o_2,...o_G$，然后通过reward model对每个输出计算当前输出的奖励，GRPO直接通过下面的公式代替广义估计函数：
$$
\hat{A}_{i, t}=\widetilde{r}_{i}=\frac{r_{i}-\operatorname{mean}(\mathbf{r})}{\operatorname{std}(\mathbf{r})},
$$
这样就不用构建一个巨大的value model。对应的，**GRPO认为经过监督微调后的SFT模型$\pi_{ref}$中已经具有一定的知识**，所以强化学习只是为了从中把知识“释放”出来，所以在下面目标函数的最后**增加了与SFT模型$\pi_{ref}$距离的惩罚项。**
$$
\begin{aligned}
\mathcal{J}_{GRPO}(\theta) & =\operatorname{E}[q\sim P(Q),\{o_{i}\}_{i=1}^{G}\sim\pi_{\theta_{old}}(O|q)] \\
 & \frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_{i}|}\sum_{t=1}^{|o_{i}|}\left\{\min\left[\frac{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q,o_{i,<t})}\hat{A}_{i,t},\mathrm{clip}\left(\frac{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q,o_{i,<t})},1-\varepsilon,1+\varepsilon\right)\hat{A}_{i,t}\right]-\beta\mathbf{D}_{KL}\left[\pi_{\theta}||\pi_{ref}\right]\right\}
\end{aligned}
$$

### RLHF(基于人类反馈的强化学习)

这一部分主要是将强化学习怎么在LLM中发挥作用的。

**提出论文**：[Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325)

**更多应用**：[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

![image-20250304154128804](https://gitee.com/fbanhua/figurebed/raw/master/images/20250304154128963.png)

上图即为PPO用于训练模型输出更符合人类喜好的结果，整个过程分三步：

- 模型输出成对的结果，人工选取最佳结果
- 从人类反馈中学习奖励模型，用于判断哪个输出的结果更好
- 针对奖励模型优化策略函数（LLM）

![image-20250304154914254](https://gitee.com/fbanhua/figurebed/raw/master/images/20250304154914422.png)

在引用的第二篇论文中，训练步骤为

- 人类编写一些回答来对模型进行监督微调
- 模型输出多个结果，人类对模型输出结果的好坏进行排序，并基于该结果训练奖励模型
- 使用奖励模型训练LLM
>>>>>>> 8f75f32 (local update)
