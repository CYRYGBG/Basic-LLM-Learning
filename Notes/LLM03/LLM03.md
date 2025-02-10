> 希望尽可能将自己学习过程中参考过的资料进行系统的整理，方便后面的初学者更快的找到自己想要的资料！

**笔记持续更新中......**

[LLM基础学习01：LLM解码策略和显存占用计算](https://zhuanlan.zhihu.com/p/21348048780)

[LLM基础学习02：分布式训练核心架构与多级并行策略详解——DDP/FSDP/ZeRO实战代码、显存优化方案及技术资源全景索引](https://zhuanlan.zhihu.com/p/21784954155)

# 简介

LLM的训练中大致可以分为六个任务：data preparation(数据准备), pre-training(预训练), fine-tuning(微调), instruction-tuning(指令调优), preference alignment(偏好对齐), and applications（下游应用）。关于数据准备和预训练阶段，通常需要进行大量的准备工作和充足的显卡资源，作者作为个人爱好学习者目前是接触不到的；而微调和指令微调的区别在于：**微调希望模型适应特定任务（比如分类，翻译等）通常是针对单一任务优化，而指令微调则希望模型可以理解并遵循自然语言表示的指令，可以完成多种任务**；偏好对齐则是训练模型使其生成更符合人类偏好的答案。

这篇笔记主要是对指令微调进行学习和代码实现。

# LLM数据集中的JSON

训练过程中数据集和数据集信息文件都会涉及到JSON文件，所以这里对用到的JSON的语法进行简要的记录，更详细的介绍和**各种数据集的用途**可以依据[LLaMA Factory](https://llamafactory.readthedocs.io/zh-cn/latest/index.html)的文档介绍，附上相关数据处理的链接：[数据集格式](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/data_preparation.html)。下面只对数据集中用到的相关语法进行介绍，主要参考资料是[菜鸟教程](https://www.runoob.com/json/json-syntax.html)。

JSON中的对象用 `{}`表示，具体应用类似于python中的字典（**一个对象中可以有多个键值对，数据由英文逗号进行分隔**），比如数据集中的一个样本就是一个对象，可以表示为：

```json
{
    "instruction": "你是一个AI助手，请准确回答下面的问题：",
    "input": "在Mygo中最后睦给soyo送的礼物是什么？",
    "output": "在《BanG Dream! It's MyGO!!!!!》中，最后睦送给Soyo的礼物是自己亲手种的小黄瓜。"
}
```

在这里，相当于一个训练的样本里有3个键值对：instruction，input和output，分别代表人类指令、人类输出和模型回答（现在只针对JSON的格式进行介绍，更详细的解释会放在后面）。另外，JSON中**没有缩进要求**，但是为了可读性还是尽量在config类的里面保持缩进比较好。

而JSON中的数组则用 `[]`表示，中间可以包含多个对象，比如：

```json
[
  {
    "instruction": "你是一个AI助手，请准确回答下面的问题：",
    "input": "在Mygo中最后睦给soyo送的礼物是什么？",
    "output": "在《BanG Dream! It's MyGO!!!!!》中，最后睦送给Soyo的礼物是自己亲手种的小黄瓜。"
	},
  {
    "instruction": "你是一个番剧高手，请准确回答下面的问题：",
    "input": "ave mujica的op是什么？",
    "output": "《BanG Dream! Ave Mujica》的OP（片头曲）是《KiLLKiSS》。"
	}
]
```

这个例子里面一个数组包含了两个样本，每个样本里都是相同的格式即包含3个键值对。

# 模型微调

下面将使用[LLaMA Factory](https://llamafactory.readthedocs.io/zh-cn/latest/index.html)作为微调框架，选取[Qwen/Qwen2.5-1.5B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct)作为base model进行微调，选取[GSM8K](hhttps://huggingface.co/datasets/openai/gsm8k)作为微调数据集。

## 数据集介绍及下载

GSM8K数据集是一个包含了8.5K道小学数学题的数据集，其中解决方案以自然语言的推理过程和最终答案构成。

这里数据使用使用huggingface进行下载，如果连接不到huggingface可以像下面的代码一样设置镜像站：

```python
import  os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset
cache_dir = r'your_path'    # 后续所有your_path需要转换为自己的路径
ds = load_dataset("openai/gsm8k", "main",cache_dir=cache_dir)
```

首先输出观察数据集的结构和其中的键值对如下，可以看到是一个`DatasetDict`中包含了`train`和`test`两个字典，一个样本中对应的两个键：`question`即为是模型的输入，`answer`即为希望得到的模型输出。

![image-20250209202645760](https://gitee.com/fbanhua/figurebed/raw/master/images/20250209202645800.png)

那么按照前文引用过的[数据集格式介绍](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README_zh.md)，需要新建一个`dataset_info.json`来对数据集信息进行描述，并且需要放置在`gsm8k-train.arrow`文件所在的同一文件夹下（和`gsm8k-test.arrow`也是同一个文件夹）：

```json
{
  "gsm8k_math_train": {
    "file_name": "gsm8k-train.arrow",
    "formatting": "alpaca",
    "columns": {
      "query": "question",
      "response": "answer"
    }
  },
  
  "gsm8k_math_test": {
    "file_name": "gsm8k-test.arrow",
    "formatting": "alpaca",
    "columns": {
      "query": "question",
      "response": "answer"
    }
  }
}

```

进一步输出训练集中前三个样本观察如下，`###`符号后接的即为最终答案。

![image-20250209202246490](https://gitee.com/fbanhua/figurebed/raw/master/images/20250209202246582.png)



## 模型介绍及下载

[Qwen/Qwen2.5-1.5B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct)是Qwen2.5系列中的一个参数比较小的decoder-only模型（从名字知道只有1.5B），可以在作者的笔记本上使用LoRA微调（8G显存）。

这里提供两种下载方法，由于huggingface下载速度比较慢，所以推荐安装[ModelScope](https://www.modelscope.cn/docs/intro/quickstart)进行下载，所以这里在命令行中使用命令

```shell
modelscope download --model Qwen/Qwen2.5-1.5B-Instruct --local_dir your_path/Qwen2.5-1.5B-Instruct
```

```shell
huggingface-cli download --resume-download Qwen/Qwen2.5-1.5B-Instruct --local-dir your_path/Qwen2.5-1.5B-Instruct
```

下载模型权重到指定文件夹（特别指出，**官方的示例代码在cmd中运行会报错**，需要把单引号去掉才能运行成功！！！）

![image-20250208230050858](https://gitee.com/fbanhua/figurebed/raw/master/images/20250208230054280.png)

## LLaMA-Factory微调

首先按照[LLaMA-Factory安装](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/installation.html#llama-factory)中的流程完成微调框架的安装，然后将新建一个文件`llama_train.yaml`并把下面的训练相关参数复制进去。然后就可以在该文件同一路径下或者指定绝对路径来使用命令进行模型训练了。

```shell
llamafactory-cli train llama_train.yaml     
```

下面是具体的**训练参数配置文件内容**，需要把里面**所有的文件路径设置为自己的文件路径**（作者这里在AutoDL上租了一张4090进行学习，所以后面路径中出现的都是作者自己的文件路径，需要进行设置）。

```yaml
# ------------------- 基础模型配置 -------------------
model_name_or_path: /root/autodl-tmp/model/Qwen2.5-1.5B-Instruct  # 使用Qwen2.5-1.5B基座模型

# ------------------- 训练阶段配置 -------------------
stage: sft
do_train: true
report_to: tensorboard    # Tensorboard设置
logging_dir: ./log_output/qwen1.5b_instruct_gsm8k_full_llamafactory 
finetuning_type: lora  # lora微调
lora_target: all
lora_rank: 16
flash_attn: fa2


# ------------------- 数据集配置 -------------------
dataset_dir: /root/autodl-tmp/dataset/openai___gsm8k/main/0.0.0/e53f048856ff4f594e959d75785d2c2d37b678ee 
dataset: gsm8k_math_train               # 对应JSON中定义的数据集名称
max_samples: null  # null表示使用全部数据，如需部分调试可设为具体数值
template: qwen  # 必须使用Qwen对应的模板格式
cutoff_len: 1024  # 上下文最大长度（GSM8K数学题通常不超过此长度）
overwrite_cache: true  # 重新预处理数据时强制刷新缓存
preprocessing_num_workers: 16  # 数据预处理并行进程数（根据CPU核数调整）

# ------------------- 训练输出相关 -------------------
output_dir: ./output/qwen1.5b_instruct_gsm8k_lora_llamafactory  # 输出目录需要存在可写权限
logging_steps: 10  # 每10步输出一次日志
save_steps: 100  # 每500步保存一次检查点
plot_loss: true  # 绘制训练损失曲线

# ------------------- 训练超参数 -------------------
per_device_train_batch_size: 8  
gradient_accumulation_steps: 4              # 梯度累积步数（等效总batch_size=2*4=8）
learning_rate: 3.0e-5                       # 1.5B模型SFT建议学习率（高于7B但低于large模型）
num_train_epochs: 6                         # GSM8K需更多epoch学习推理逻辑
max_grad_norm: 0.5                          # 梯度裁剪阈值
lr_scheduler_type: cosine                   
warmup_ratio: 0.15                          # warmup阶段占训练总步数的比例
weight_decay: 0.05                          # 新增权重衰减，防止过拟合

# ------------------- 验证与评估 -------------------
val_size: 0.1  # 10%数据作为验证集
per_device_eval_batch_size: 16  # 评估时batch_size可以更大
eval_strategy: steps  # 按步数评估
eval_steps: 100  # 每200步验证一次（GSM8K需要及时评估推理能力）

# ------------------- 显存优化 -------------------
gradient_checkpointing: true  # 激活梯度检查点节省显存
optim: adamw_torch  # 推荐使用AdamW优化器

```

可以使用`tensorboard`查看训练效果，根据上述文件配置的文件记录路径，可以使用以下命令：

```shell
tensorboard --port 6007 --logdir /root/autodl-tmp/llm03/log_output/qwen1.5b_instruct_gsm8k_full_llamafactory
```

![image-20250210205453623](https://gitee.com/fbanhua/figurebed/raw/master/images/20250210205453705.png)

可以看到，大概在迭代600step后模型就会在训练集上过拟合（验证集损失明显上升），同时由于上面的训练命令中设置每200步保存一次模型参数，所以我们选取**过拟合前的模型参数**（即600step时保存的参数，具体保存参数的列表可以看下图）作为最终测试的参数。

![image-20250210205654816](https://gitee.com/fbanhua/figurebed/raw/master/images/20250210205654885.png)

微调结束后，由于使用的是LoRA训练，所以需要**对模型参数进行合并导出**，根据[官方导出教程](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/merge_lora.html)，合并命令和配置文件`merge_config.yaml`如下：

```shell
llamafactory-cli export merge_config.yaml
```

```yaml
### model
### 选取eval loss最小的checkpoint进行合并
model_name_or_path: /root/autodl-tmp/model/Qwen2.5-1.5B-Instruct
adapter_name_or_path: /root/autodl-tmp/llm03/output/qwen1.5b_instruct_gsm8k_lora_llamafactory/checkpoint-600
template: qwen
finetuning_type: lora

### export
export_dir: /root/autodl-tmp/model/Qwen2.5-1.5B-Instruct-lora-sft
export_size: 1
export_device: cpu
export_legacy_format: false
```

# 模型评估

## 评估工具

现在使用一个评估工具：[Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)来对模型的效果进行评价，有一个介绍的比较好的[安装和使用教程](https://medium.com/@cch.chichieh/llm-%E8%A9%95%E4%BC%B0%E6%95%99%E5%AD%B8-eleutherai-lm-evaluation-harness-42628a4362f7#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6ImVlYzUzNGZhNWI4Y2FjYTIwMWNhOGQwZmY5NmI1NGM1NjIyMTBkMWUiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJhenAiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJhdWQiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJzdWIiOiIxMDkwMDM0Mzg1OTIzNTMyMDQwOTUiLCJlbWFpbCI6ImZiYW5odWExMTIzQGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJuYmYiOjE3MzkwMDkxNjIsIm5hbWUiOiJubiBubiIsInBpY3R1cmUiOiJodHRwczovL2xoMy5nb29nbGV1c2VyY29udGVudC5jb20vYS9BQ2c4b2NMSUxuYXgwUS1Kd2NLVUstNlJNZnBwZ29yU1A0U19jdHFub08yWXRSRmtxV2ZVc0ZJPXM5Ni1jIiwiZ2l2ZW5fbmFtZSI6Im5uIiwiZmFtaWx5X25hbWUiOiJubiIsImlhdCI6MTczOTAwOTQ2MiwiZXhwIjoxNzM5MDEzMDYyLCJqdGkiOiJhMWIwNzY3N2EwMWU2OTU3ZTE2MWVhMWE1NGUwOGE3ZjE3NmU0ZGYwIn0.oC9JTA1bLjqtrN9YV16Qt9w86OCCkAsWF84nW7O0cfTQ-lbv4OGj1SfXfsPRzHvaS5V4eNPL2_ZhL9VM6wPiXDwUpqcPXe7YAnSfugeJdaRDOlI8a6TbAhFsle2Qn3Dq9BdnD2aJERIe3rgbZ04Vb_uChI5AvRvsieKLOwpy_sHVxDbuViSkcOw-eM-aCDbM3CRHKwYqWA6WkF2GJG3-CCbzuKNadDJhfO0BPThZA7cqce7lXDq8olJADXVVL8Wv8T95FqEQZVBxcONhtSnMGzbPyFb-ksBudkS6LGREzhR1MGgHy8qDeKM8m-D1ZCkij7VJoq1XB4h8Vq5VzLn9MA)。

使用下面的命令**对指定路径下的模型权重文件进行一次测试**。

```shell
lm_eval --model hf --model_args pretrained="/root/autodl-tmp/model/Qwen2.5-1.5B-Instruct" --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./eval_results --num_fewshot 0
```

其中指定了`num_fewshot`参数为每个问题**提供指定个数的示例**（可以理解为做题前可以查看例题的解题格式，由于训练时模型是没有例题的，所以这里设置为0）。

如果要**使用多卡进行评估**，则需要使用`accelerate launch`，具体的命令如下(该命令会在所有的可用GPU上进行推理)

```shell
accelerate launch -m lm_eval --model hf --model_args pretrained="/root/autodl-tmp/model/Qwen2.5-1.5B-Instruct" --tasks gsm8k --batch_size 8 --output_path ./eval_results --num_fewshot 0
```

## 本地数据集

如果要使用本地模型和调用**本地**数据集，则在对应的库（这里是`\lm-evaluation-harness\lm_eval\tasks\gsm8k`）下找到文件`gsm8k.yaml`后复制其中的内容得到`gsm8k-local.yaml`文件**在相同的文件夹**，然后在里面根据[官方教程](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md#using-local-datasets)进行了修改，修改的内容如下所示（实际上不规范啊，不能验证集和测试集是同一个的，这里是图方便没在训练集里面划分验证集直接做个示范）：

![image-20250209131047499](https://gitee.com/fbanhua/figurebed/raw/master/images/20250209131047556.png)

作者在自己电脑基于这个配置文件使用下面的命令进行了测试，是可以运行成功的。

```shell
lm_eval --model hf --model_args pretrained="D:\04.Code\model\Qwen2.5-1.5B-Instruct" --tasks gsm8k-local --device cuda:0 --batch_size 1 --output_path ./eval_results --trust_remote_code
```

## 评估结果

首先使用**未经训练的模型进行评估**，测试完之后可以得到以下结果。可以看到，这个模型的能力还是挺菜的（n-shot为5表示了每道题会提供0个例题，flexible-extract表示从输出中灵活提取数字答案，strict-match表示必须严格匹配答案格式，可以看出模型的**规范输出能力很差**，但基本还是能得到一些正确结果）。

| Tasks | Version | Filter           | n-shot | Metric      |      |  Value |      | Stderr |
| ----- | ------: | ---------------- | -----: | ----------- | ---- | -----: | ---- | -----: |
| gsm8k |       3 | flexible-extract |      0 | exact_match | ↑    | 0.1494 | ±    | 0.0098 |
|       |         | strict-match     |      0 | exact_match | ↑    | 0.0000 | ±    | 0.0000 |

下面是**微调后**的结果，可以看到微调对模型的提升是很大的，而且严格匹配答案格式，但是目前还没搞清楚为什么宽松模式的得分会比严格模式的低那么多。

| Tasks | Version | Filter           | n-shot | Metric      |      |  Value |      | Stderr |
| ----- | ------: | ---------------- | -----: | ----------- | ---- | -----: | ---- | -----: |
| gsm8k |       3 | flexible-extract |      0 | exact_match | ↑    | 0.3806 | ±    | 0.0134 |
|       |         | strict-match     |      0 | exact_match | ↑    | 0.4723 | ±    | 0.0138 |
