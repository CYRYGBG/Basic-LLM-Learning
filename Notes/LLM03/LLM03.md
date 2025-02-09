# TODO

- [x] 在gsm8k上不训练测试一次模型
- [ ] 查看peft怎么选取超参数以及lora的超参数是否能迁移到全参训练上
- [ ] 设置好autodl的训练环境并进行测试（捋清楚所有文件和流程）
- [ ] 使用llama factory训练模型进行测试选取超参数
- [ ] 使用Swift和llama factory分别训练一个模型，记录峰值显存和训练时间
- [ ] 对比不同的分布式训练方法的峰值显存和训练时间（可选）
- [ ] 学习超参数选取方法（可选）

# 简介

LLM的训练中大致可以分为六个任务：data preparation(数据准备), pre-training(预训练), fine-tuning(微调), instruction-tuning(指令调优), preference alignment(偏好对齐), and applications（下游应用）。关于数据准备和预训练阶段，通常需要进行大量的准备工作和充足的显卡资源，作者作为个人爱好学习者目前是接触不到的；而微调和指令微调的区别在于：**微调希望模型适应特定任务（比如分类，翻译等）通常是针对单一任务优化，而指令微调则希望模型可以理解并遵循自然语言表示的指令，可以完成多种任务**；偏好对齐则是训练模型使其生成更符合人类偏好的答案。

这篇笔记主要是对指令微调进行学习和代码实现。

# LLM数据集中的JSON

更详细的介绍和**各种数据集的用途**可以依据[LLaMA Factory](https://llamafactory.readthedocs.io/zh-cn/latest/index.html)的文档介绍，附上相关数据处理的链接：[数据集格式](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/data_preparation.html)。下面只对数据集中用到的相关语法进行介绍，主要参考资料是[菜鸟教程](https://www.runoob.com/json/json-syntax.html)。

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

下面将分别使用[LLaMA Factory](https://llamafactory.readthedocs.io/zh-cn/latest/index.html)和[Swift](https://swift.readthedocs.io/zh-cn/latest/index.html)作为微调框架，选取[Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)作为base model进行微调，选取[GSM8K](hhttps://huggingface.co/datasets/openai/gsm8k)作为微调数据集。

## 数据集介绍及下载

GSM8K数据集是一个包含了8.5K道小学数学题的数据集，其中解决方案以自然语言的推理过程和最终答案构成。

这里数据使用使用huggingface进行下载，如果连接不到huggingface可以像下面的代码一样设置镜像站：

```python
import  os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset
cache_dir = r'/root/autodl-tmp/dataset'  # 作者在AutoDL租了显卡进行学习，所以后续出现该路径可以转换为自己的路径
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

Qwen/Qwen2.5-1.5B是Qwen2.5系列中的一个参数比较小的decoder-only模型（从名字知道只有1.5B），可以在作者的笔记本上使用LoRA微调（8G显存），同时也是比较强的一个基础模型（可以在后面中不训练直接测试的结果看出）。

由于huggingface下载网络比较差，所以这里安装[ModelScoep](https://www.modelscope.cn/docs/intro/quickstart)下载，所以这里在命令行中使用命令

```shell
modelscope download --model Qwen/Qwen2.5-1.5B --local_dir /root/autodl-tmp/model/Qwen2.5-1.5B
```

下载模型权重到指定文件夹（特别指出，**官方的示例代码在cmd中运行会报错**，需要把单引号去掉才能运行成功！！！）

![image-20250208230050858](https://gitee.com/fbanhua/figurebed/raw/master/images/20250208230054280.png)

## 微调

首先按照[LLaMA-Factory安装](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/installation.html#llama-factory)中的流程完成微调框架的安装，然后将新建一个文件`llama_train.yaml`并把下面的训练相关参数复制进去。然后就可以在该文件同一路径下或者指定绝对路径来使用命令`llamafactory-cli train .\llama_train.yaml`进行模型训练了。

# 模型评估

## 评估工具及base model测试

现在使用一个评估工具：[Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)来对模型的效果进行评价，有一个介绍的比较好的[教程](https://medium.com/@cch.chichieh/llm-%E8%A9%95%E4%BC%B0%E6%95%99%E5%AD%B8-eleutherai-lm-evaluation-harness-42628a4362f7#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6ImVlYzUzNGZhNWI4Y2FjYTIwMWNhOGQwZmY5NmI1NGM1NjIyMTBkMWUiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJhenAiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJhdWQiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJzdWIiOiIxMDkwMDM0Mzg1OTIzNTMyMDQwOTUiLCJlbWFpbCI6ImZiYW5odWExMTIzQGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJuYmYiOjE3MzkwMDkxNjIsIm5hbWUiOiJubiBubiIsInBpY3R1cmUiOiJodHRwczovL2xoMy5nb29nbGV1c2VyY29udGVudC5jb20vYS9BQ2c4b2NMSUxuYXgwUS1Kd2NLVUstNlJNZnBwZ29yU1A0U19jdHFub08yWXRSRmtxV2ZVc0ZJPXM5Ni1jIiwiZ2l2ZW5fbmFtZSI6Im5uIiwiZmFtaWx5X25hbWUiOiJubiIsImlhdCI6MTczOTAwOTQ2MiwiZXhwIjoxNzM5MDEzMDYyLCJqdGkiOiJhMWIwNzY3N2EwMWU2OTU3ZTE2MWVhMWE1NGUwOGE3ZjE3NmU0ZGYwIn0.oC9JTA1bLjqtrN9YV16Qt9w86OCCkAsWF84nW7O0cfTQ-lbv4OGj1SfXfsPRzHvaS5V4eNPL2_ZhL9VM6wPiXDwUpqcPXe7YAnSfugeJdaRDOlI8a6TbAhFsle2Qn3Dq9BdnD2aJERIe3rgbZ04Vb_uChI5AvRvsieKLOwpy_sHVxDbuViSkcOw-eM-aCDbM3CRHKwYqWA6WkF2GJG3-CCbzuKNadDJhfO0BPThZA7cqce7lXDq8olJADXVVL8Wv8T95FqEQZVBxcONhtSnMGzbPyFb-ksBudkS6LGREzhR1MGgHy8qDeKM8m-D1ZCkij7VJoq1XB4h8Vq5VzLn9MA)。使用下面的命令对未经训练的模型进行一次测试。

```shell
lm_eval --model hf --model_args pretrained="/root/autodl-tmp/model/Qwen2.5-1.5B" --tasks gsm8k --device cuda:0 --batch_size auto --output_path ./eval_results
```

也可以指定`num_fewshot`参数为每个问题提供指定个数的示例（可以理解为做题前可以查看例题的解题格式），相关的代码就是在最后指定相关参数：

```shell
lm_eval --model hf --model_args pretrained="/root/autodl-tmp/model/Qwen2.5-1.5B" --tasks gsm8k --device cuda:0 --batch_size auto --output_path ./eval_results --num_fewshot 1
```

如果要使用多卡进行评估，则需要使用`accelerate launch`，具体的命令如下(该命令会在所有的可用GPU上进行推理)

```shell
accelerate launch -m lm_eval --model hf --model_args pretrained="/root/autodl-tmp/model/Qwen2.5-1.5B" --tasks gsm8k --batch_size auto --output_path ./eval_results
```

测试完之后可以得到以下结果，可以看到这个base model的能力还是挺菜的（n-shot为5表示了每道题会提供5个例题，flexible-extract表示从输出中灵活提取数字答案，strict-match表示必须严格匹配答案格式）。

| Tasks | Version | Filter           | n-shot | Metric      |      |  Value |      | Stderr |
| ----- | ------: | ---------------- | -----: | ----------- | ---- | -----: | ---- | -----: |
| gsm8k |       3 | flexible-extract |      5 | exact_match | ↑    | 0.6224 | ±    | 0.0134 |
|       |         | strict-match     |      5 | exact_match | ↑    | 0.6179 | ±    | 0.0134 |
| gsm8k |       3 | flexible-extract |      1 | exact_match | ↑    | 0.2123 | ±    | 0.0113 |
|       |         | trict-match      |      1 | exact_match | ↑    | 0.0045 | ±    | 0.0019 |

## 本地数据集

如果要使用本地模型和调用**本地**数据集，则在对应的库（这里是`\lm-evaluation-harness\lm_eval\tasks\gsm8k`）下找到文件`gsm8k.yaml`后复制其中的内容得到`gsm8k-local.yaml`文件**在相同的文件夹**，然后在里面根据[官方教程](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md#using-local-datasets)进行了修改，修改的内容如下所示（实际上不规范啊，不能验证集和测试集是同一个的，这里是图方便没在训练集里面划分验证集直接做个示范）：

![image-20250209131047499](https://gitee.com/fbanhua/figurebed/raw/master/images/20250209131047556.png)

基于这个配置文件使用下面的命令进行了测试，是可以运行成功的。

```shell
lm_eval --model hf --model_args pretrained="D:\04.Code\model\Qwen2.5-1.5B" --tasks gsm8k-local --device cuda:0 --batch_size 1 --output_path ./eval_results --trust_remote_code
```

