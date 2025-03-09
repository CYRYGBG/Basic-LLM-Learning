# -*- coding: utf-8 -*-
# @Time    : 2025/03/08 15:42:38
# @Author  : Chen, Y.R.
# @File    : FSDP.py
# @Software: VSCode
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 禁用所有 TensorFlow 日志
os.environ["NCCL_P2P_DISABLE"] = "1" 
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"  
os.environ["MASTER_ADDR"] = "localhost"  
os.environ["MASTER_PORT"] = "29501"     


import warnings
warnings.filterwarnings("ignore")
from tqdm.auto import tqdm
import random
import copy
import re
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os
import functools
import torch.distributed as dist  # 用于多线程数据的处理
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, BackwardPrefetch
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, enable_wrap, wrap
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer



def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(42)

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
And must have '####' before the final answer number.
"""


def extract_answer_from_model_output(text):
   # Split on <answer> and take everything after the last occurrence
   parts = text.split("<answer>")
   # 如果没有找到<answer>，那就是没有进行切割，长度只有1
   if len(parts) < 2:  
       return None
   last_part = parts[-1]  # 只取最后一个<answer>部分

   # 结尾没有</answer>
   if "</answer>" not in last_part:
       return None
   answer = last_part.split("</answer>")[0].strip()
   return None if answer == "..." else answer


def extract_answer_from_dataset(text):
   if "####" not in text:
       return None
   return text.split("####")[1].strip()


def prepare_dataset(split="train"):
    data = load_dataset('openai/gsm8k', 'main', cache_dir="./dataset", download_mode="reuse_dataset_if_exists")[split]
    formatted_data = []

    tmp = data[0]
    print("-----------打印示例数据----------")
    print("question: \n", tmp["question"])
    print("answer: \n", tmp["answer"])

    for example in data:
        # 将消息列表转换为单个字符串
        prompt_str = build_prompt([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]}
        ])
        formatted_example = {
            "prompt": prompt_str,  # Now a string rather than a list.
            "answer": extract_answer_from_dataset(example["answer"])
        }
        formatted_data.append(formatted_example)
    return formatted_data


def build_prompt(messages):
   return "\n".join([msg["content"].strip() for msg in messages])


# 从文本中提取最后一个数字
def extract_last_number(text):
   # 清理文本中的特殊符号：移除美元符号($)和百分号(%)
   text = text.replace('$', '').replace('%', '')
   # 正则表达式模式匹配行尾的数值（支持负数、小数、空格和等号前缀）
   pattern = r'(?:^|\s|=)\s*(-?\d*\.?\d+)\s*$'
   # 在文本中搜索匹配模式的内容
   match = re.search(pattern, text)
   # 返回提取的数值（浮点数），若未匹配则返回None
   return float(match.group(1)) if match else None


# 只有一个数字的情况下使用，其余情况返回None
def extract_single_number(text):
   numbers = re.findall(r'-?\d*\.?\d+', text)
   return float(numbers[0]) if len(numbers) == 1 else None


def evaluate_model(model, tokenizer, eval_examples, device):
   # 将模型设置为评估模式
   model.eval()
   correct = 0
   total = len(eval_examples)
   print("\n" + "="*50)
   print("EVALUATION ON", total, "EXAMPLES")
   print("="*50)

   for example in eval_examples:
       # Get the prompt and expected answer
       full_prompt = example["prompt"]
       expected = example["answer"]

       # Tokenize and generate response
       inputs = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
       with torch.no_grad():
           outputs = model.generate(
               inputs,                                      # 输入张量
               max_new_tokens=512,                          # 最大生成 token 数量
               temperature=0.7,                             # 控制生成文本的随机性
               num_return_sequences=1,                      # 只生成一个输出序列
               pad_token_id=tokenizer.pad_token_id,         # 填充 token 的 ID
               eos_token_id=tokenizer.eos_token_id,         # 结束 token 的 ID
               forced_eos_token_id=tokenizer.eos_token_id,  # 强制在生成结束时添加结束 token
               early_stopping=False,                        # 不启用早停机制（生成固定长度）
           )
       # 将生成的 token 序列解码为可读文本
       response = tokenizer.decode(outputs[0], skip_special_tokens=True)

       try:
           # Extract answer and check correctness
           predicted = extract_answer_from_model_output(response)

           # Try different matching methods
           if predicted == expected:  # 完全精确匹配
               is_correct = True
           else:
               # 尝试提取单个数值进行匹配
               pred_num = extract_single_number(str(predicted)) # 从模型答案中提取数值
               exp_num = extract_single_number(str(expected))   # 从正确答案中提取数值
               if pred_num is not None and exp_num is not None and pred_num == exp_num:
                   is_correct = True
               else:
                   # 尝试提取最后一个数值进行匹配
                   pred_num = extract_last_number(str(predicted))   # 从模型答案中提取数值    
                   exp_num = extract_last_number(str(expected))     # 从正确答案中提取数值
                   is_correct = (pred_num is not None and exp_num is not None and
                               pred_num == exp_num)

           # 有一个对就算对
           if is_correct:
               correct += 1

       except Exception as e:
           print("\nFailed to parse model output for prompt:")
           print(full_prompt)
           print("Error:", e)
           print("-"*50)

   # Calculate and print final accuracy
   accuracy = (correct / total) * 100
   print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
   print("="*50)

   # Return model to training mode
   model.train()
   return accuracy


# 准确性奖励(精准匹配2分, 答案匹配1.5分, 其余0分)
def correctness_reward(prompts, completions, answer, **kwargs):
    # 从模型输出中提取<answer>和</answer>之间的答案
    responses = [completion[0]['content'] for completion in completions]
    extracted = [extract_answer_from_model_output(r) for r in responses]
    rewards = []
    for r, a in zip(extracted, answer):
        # 完全匹配给2分
        if r == a:  # Exact match case
            rewards.append(2.0)
        # 不完全匹配给1.5分（数值答案正确）
        else:    
            # Try numeric equivalence
            r_num = extract_single_number(str(r))
            a_num = extract_single_number(str(a))
            if r_num is not None and a_num is not None and r_num == a_num:
                rewards.append(1.5)
            else:
                rewards.append(0.0)
   # Log completion lengths
    completion_lengths = [len(response.split()) for response in responses]
    return rewards

# 格式奖励(关于<reasoning> <reasoning> <answer> <answer>)
def format_reward(completions, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    format_scores = []
    for response in responses:
        score = 0.0
        if "<reasoning>" in response: score += 0.2
        if "</reasoning>" in response: score += 0.2
        if "<answer>" in response: score += 0.2
        if "#### " in response: score += 0.2
        if "</answer>" in response: 
            score += 0.2
            # 冗余输出惩罚
            answer_end = response.find("</answer>") + len("</answer>")
            extra_text = response[answer_end:].strip()
            score = score * 0.5 if extra_text else score   # 如果"</answer>"后面还有别的输出，则格式分得分折半
        rewards.append(score)
        format_scores.append(score)
    return rewards

# 计算总得分
def combined_reward(prompts, completions, answer):
   # Get individual rewards
   correctness_scores = correctness_reward(prompts=prompts, completions=completions, answer=answer)
   format_scores = format_reward(completions=completions)

   # Combine rewards - correctness is weighted more heavily
   combined_rewards = []
   for c_score, f_score in zip(correctness_scores, format_scores):
       # Correctness score range: 0.0 to 2.0
       # Format score range: 0.0 to 1.0
       # Total range: 0.0 to 2.8
       combined_rewards.append(c_score + f_score)

   return combined_rewards


def selective_log_softmax(logits, input_ids):
    """
    计算词汇表中特定 token 的对数概率。

    参数:
        logits (torch.Tensor): 模型输出的原始 logits（未归一化）。
        input_ids (torch.Tensor): 需要获取对数概率的目标 token ID。

    返回:
        torch.Tensor: 选中 token 的对数概率，形状与 `input_ids` 一致。

    实现步骤:
        1. 对 logits 应用 log_softmax，得到词汇表维度上的对数概率分布。
        2. 使用 `gather` 提取与 input_ids 对应的特定 token 的对数概率。
        3. 压缩多余的维度，使输出形状与 input_ids 匹配。
    """
    # 对 logits 应用 log_softmax（沿词汇维度归一化）
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    
    # 从 log_probs 中提取 input_ids 对应的值，并压缩最后一个维度
    return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)


def compute_log_probs(model, input_ids, attention_mask, logits_to_keep):
    # 获取模型输出 logits，并移除最后一个 token 的预测（形状变为 [batch_size, seq_len-1, vocab_size]）
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    
    # 截取 input_ids 的最后 logits_to_keep 个 token（目标 token）
    input_ids = input_ids[:, -logits_to_keep:]
    
    # 截取 logits 的最后 logits_to_keep 个位置（对应目标 token 的预测）
    logits = logits[:, -logits_to_keep:, :]
    
    # 计算目标 token 的对数概率
    return selective_log_softmax(logits, input_ids)


def create_completion_mask(completion_ids, eos_token_id):
    # 标记 EOS token 的位置（布尔张量）
    is_eos = completion_ids == eos_token_id

    # 初始化默认 EOS 索引为序列长度（若无 EOS，则保留全部 token）
    eos_idx = torch.full(
        (is_eos.size(0),), 
        is_eos.size(1),  # 默认值为序列长度 seq_len
        dtype=torch.long, 
        device=completion_ids.device
    )

    # 检查每个序列是否存在至少一个 EOS
    mask_exists = is_eos.any(dim=1)

    # 对有 EOS 的序列，获取首个 EOS 的位置索引
    # （利用 argmax 返回第一个 True 的位置）
    eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]

    # 生成位置索引矩阵 [batch_size, seq_len]
    # 例如：若 seq_len=3，则每行为 [0, 1, 2]
    sequence_indices = torch.arange(
        is_eos.size(1), 
        device=completion_ids.device
    ).expand(is_eos.size(0), -1)

    # 生成掩码：位置索引 <= 首个 EOS 索引的位置设为 1
    return (sequence_indices <= eos_idx.unsqueeze(1)).int()


def generate_completions(model, tokenizer, prompts, rank, num_generations=4, max_completion_length=32):
    device = torch.device(f"cuda:{rank}") 

    inputs = tokenizer(
        prompts, 
        return_tensors="pt",  
        padding=True,         
        padding_side="left"   
    ).to(device)   # 移动到当前设备
    prompt_ids = inputs["input_ids"].to(device, non_blocking=True)
    prompt_mask = inputs["attention_mask"].to(device, non_blocking=True)
    prompt_length = prompt_ids.size(1)
    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)
    
    outputs = model.module.generate(
        input_ids=prompt_ids,
        attention_mask=prompt_mask,
        max_new_tokens=max_completion_length,  
        do_sample=True,        
        temperature=1.0,       
        pad_token_id=tokenizer.pad_token_id,   
        eos_token_id=tokenizer.eos_token_id,  
        early_stopping=False,  
        synced_gpus=True   # 跨GPU同步生成  
    )
    
    # print(f"Output batch size: {outputs.size(0)}, Device after model: {outputs.device}")
    completion_ids = outputs[:, prompt_length:]
    completion_mask = create_completion_mask(completion_ids, tokenizer.eos_token_id)
    
    return prompt_ids, prompt_mask, completion_ids, completion_mask


def generate_rollout_data(model, ref_model, tokenizer, batch_samples, num_generations, max_completion_length, rank):
    # 从训练样本中提取提示和答案
    # 支持字典和元组两种数据格式
    # 示例：
    # 'prompt': 'Respond in the following format:\n<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\nNatalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?'
    # 'answer': '72'
    prompts = [sample["prompt"] if isinstance(sample, dict) else sample[0] for sample in batch_samples]
    answers = [sample["answer"] if isinstance(sample, dict) else sample[1] for sample in batch_samples]

    # 禁用梯度计算（生成阶段）
    with torch.no_grad():
        prompt_ids, prompt_mask, completion_ids, completion_mask = generate_completions(
            model=model, 
            tokenizer=tokenizer, 
            prompts=prompts, 
            num_generations=num_generations, 
            max_completion_length=max_completion_length,
            rank=rank
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)             
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)     
        logits_to_keep = completion_ids.size(1)
        old_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)       # 当前策略模型
        ref_log_probs = compute_log_probs(ref_model, input_ids, attention_mask, logits_to_keep)   # 参考模型

    formatted_completions = [[
        {'content': tokenizer.decode(ids, skip_special_tokens=True)}  # 解码并跳过特殊token
    ] for ids in completion_ids]  # 每个生成样本包装成字典列表形式

    # 扩展原始样本以匹配生成的补全数量
    repeated_prompts = [p for p in prompts for _ in range(num_generations)]    
    repeated_answers = [a for a in answers for _ in range(num_generations)]  

    # 封装所有训练数据
    return {
        "input_ids": input_ids,                    # 完整输入序列的token ID
        "attention_mask": attention_mask,          # 对应的注意力掩码
        "completion_mask": completion_mask,        # 仅补全部分的掩码
        "old_log_probs": old_log_probs,            # 策略模型的对数概率
        "ref_log_probs": ref_log_probs,            # 参考模型的对数概率（用于KL约束）
        "formatted_completions": formatted_completions,  # 可用于奖励模型的可读文本
        "repeated_prompts": repeated_prompts,      # 扩展后的提示文本（与生成样本对齐）
        "repeated_answers": repeated_answers,      # 扩展后的答案（与生成样本对齐）
        "logits_to_keep": logits_to_keep,          # 补全文本的token长度
        "batch_size": len(prompts),                # 原始批次数目
        "num_generations": num_generations         # 每个样本生成数
    }


def grpo_loss(model, ref_model, rollout_data, tokenizer, reward_function, beta=0.01, epsilon=0.2):
    """
    计算GRPO（Generalized Reinforcement Policy Optimization）损失函数。

    Args:
        model: 当前训练的策略模型
        ref_model: 参考模型（用于KL散度计算）
        rollout_data (dict): 由generate_rollout_data生成的交互数据
        tokenizer: 文本编码/解码器
        reward_function: 奖励计算函数
        beta (float): KL散度惩罚系数（默认0.01）
        epsilon (float): PPO裁剪范围参数（默认0.2）

    Returns:
        torch.Tensor: 需要最小化的GRPO损失值
        float: 当前批次的平均奖励值

    Explanation:
        实现GRPO算法的核心损失计算，包含：
        - 策略梯度优化（PPO裁剪机制）
        - KL散度约束（防止策略过激偏离参考模型）
        - 奖励标准化处理
    """
    # 自动检测计算设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 从交互数据中提取关键元素
    input_ids = rollout_data["input_ids"]  # 完整输入序列的token ID (batch_size*num_gen, seq_len)
    attention_mask = rollout_data["attention_mask"]  # 注意力掩码
    completion_mask = rollout_data["completion_mask"]  # 补全部分的掩码（仅生成部分为1）
    logits_to_keep = rollout_data["logits_to_keep"]  # 补全文本的token长度
    old_log_probs = rollout_data["old_log_probs"]  # 旧策略的对数概率
    ref_log_probs = rollout_data["ref_log_probs"]  # 参考模型的对数概率

    # 1. 计算当前策略模型的对数概率, 梯度可追踪
    token_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)
    
    # 2. 新旧策略概率比率（指数运算将log概率转换为概率比值）
    ratio = torch.exp(token_log_probs - old_log_probs)  # 形状: (batch_size*num_gen, seq_len)

    # 3. 计算奖励值 
    rewards = torch.tensor(
        reward_function(
            prompts=rollout_data["repeated_prompts"],  # 扩展后的提示列表
            completions=rollout_data["formatted_completions"],  # 格式化的生成文本
            answer=rollout_data["repeated_answers"]  # 扩展后的标准答案
        ),    # 把模型的输出输入到奖励计算函数，即combined_reward
        dtype=torch.float32,
        device=device
    )  

    # 4. 维度调整与奖励标准化 （广义优势函数）
    batch_size = rollout_data["batch_size"]  # 原始批次数
    num_generations = rollout_data["num_generations"]  # 每个样本生成数
    
    # 将奖励重塑为矩阵形式： (batch_size, num_generations)
    rewards = rewards.view(batch_size, num_generations)
    
    # 计算并打印平均奖励（用于监控训练进度）
    avg_reward = rewards.mean().item()
    print("Average Reward:", avg_reward)
    
    # 计算每个提示对应的平均奖励和标准差（用于标准化）
    mean_rewards = rewards.mean(dim=1).repeat_interleave(num_generations)  # (batch_size*num_gen,)
    std_rewards = rewards.std(dim=1).repeat_interleave(num_generations)    
    
    # 优势函数计算（标准化处理）
    advantages = ((rewards.view(-1) - mean_rewards) / (std_rewards + 1e-4)).unsqueeze(1)  

    # 5. PPO裁剪目标计算 
    surr1 = ratio * advantages          # 未裁剪的原始目标
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages  # 裁剪后的保守目标
    surrogate_loss = torch.min(surr1, surr2)  # 取两者较小值 → 形状: (batch_size*num_gen, seq_len)

    # 6. KL散度惩罚项计算 
    # 使用近似公式：KL(p||q) ≈ exp(log_p - log_q) - (log_p - log_q) - 1
    kl = torch.exp(ref_log_probs - token_log_probs) - (ref_log_probs - token_log_probs) - 1

    # 7. 综合损失计算 
    per_token_loss = surrogate_loss - beta * kl  
    
    # 应用补全掩码（仅计算生成部分的损失）并平均
    masked_loss = per_token_loss * completion_mask  # 形状保持不变
    loss = -((masked_loss.sum(dim=1) / completion_mask.sum(dim=1)).mean())  # 负号因为要最大化奖励
    
    return loss, avg_reward


# 1. 增加FSDP相关参数设置的函数
def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


# 2. 修改模型封装方式
def get_model(model, rank):
    # 获取需要忽略的模块（embedding和lm_head）
    # ignored_modules = []
    # if hasattr(model.model, 'embed_tokens'):
    #     ignored_modules.append(model.model.embed_tokens)
    # if hasattr(model, 'lm_head'):
    #     ignored_modules.append(model.lm_head)
    
    # 使用transformer层自动包装策略
    my_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen2DecoderLayer}
    )
    
    return FSDP(
        model.to(rank),
        auto_wrap_policy=my_auto_wrap_policy,
        device_id=rank,
        use_orig_params=True,
        # ignored_modules=ignored_modules  # 关键修改：排除关键模块分片
    )

    


def train_with_grpo(model, 
                    ref_model,
                    tokenizer, 
                    train_data, 
                    rank,
                    world_size,
                    num_iterations=1, 
                    num_steps=500, 
                    batch_size=4,
                    num_generations=4, 
                    max_completion_length=128, 
                    beta=0.1,
                    learning_rate=5e-6, 
                    mu=3, 
                    epsilon=0.2, 
                    reward_function=None, 
                    device_ids=None):

    # setup(rank, world_size)
   
    if rank == 0:
        writer = SummaryWriter(log_dir="runs/dlc_FSDP_experiment")
    else:
        writer = None

    set_random_seed(42 + rank)
    # 数据分片逻辑（关键新增代码）
    def get_local_data(full_data, rank, world_size):
        # 计算分片步长
        chunk_size = len(full_data) // world_size
        # 分配数据分片
        start_idx = rank * chunk_size
        end_idx = start_idx + chunk_size if rank != world_size-1 else len(full_data)
        return full_data[start_idx:end_idx]

    # 获取当前进程的本地数据分片
    local_train_data = get_local_data(train_data, rank, world_size)
    # print(f"Rank {rank} received {len(local_train_data)} samples")

    for iteration in range(num_iterations):
        if rank == 0:
            print(f"\n当前迭代 {iteration+1}/{num_iterations}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        model.train()  

        for step in range(num_steps):
            # batch_samples = random.sample(train_data, batch_size)
            batch_samples = random.sample(local_train_data, batch_size)
            
            # 生成交互数据（禁用梯度计算）
            with torch.no_grad():
                rollout_data = generate_rollout_data(
                    model=model,  
                    ref_model=ref_model,
                    tokenizer=tokenizer,
                    batch_samples=batch_samples,
                    num_generations=num_generations,
                    max_completion_length=max_completion_length,
                    rank=rank
                )
            
            for grpo_iter in range(mu):
                loss, avg_reward = grpo_loss(
                    model,
                    ref_model,
                    rollout_data,
                    tokenizer,
                    reward_function,
                    beta=beta,
                    epsilon=epsilon
                )
                
                # 梯度更新步骤
                optimizer.zero_grad()  # 清空梯度
                loss.backward()  # 反向传播
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)  # 梯度裁剪
                optimizer.step()  # 参数更新
                
                global_step = iteration * num_steps + step  # 需要计算全局步数
                avg_loss = torch.tensor(loss.item(), device=f"cuda:{rank}")
                dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
                if rank == 0:
                    writer.add_scalar('Loss/train', avg_loss.item(), global_step)
                    writer.add_scalar('Reward/train', avg_reward, global_step)
                
                # 打印训练进度
                if rank == 0:
                    print(f"迭代 [{iteration+1}/{num_iterations}], 步骤 [{step+1}/{num_steps}], "
                        f"GRPO更新 [{grpo_iter+1}/{mu}], 损失: {loss.item():.4f}")
                    
                    
                    # GPU监控（可选）
                    for i in range(torch.cuda.device_count()):
                        print(f"GPU {i} 内存使用: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MiB, "
                                f"利用率: {torch.cuda.utilization(i)}%")
                        print("------------------------------------------------------------------------", end="\n")
    
    # 训练完成关闭 writer
    if rank == 0:
        writer.close()

    # 返回原始模型（解除DataParallel包装）
    return model.module


def main_fn(rank, world_size, train_data, eval_data, base_model_path, training_config):
    # 初始化进程组
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    # 调试：确认设备设置
    print(f"Rank {rank}: CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}, "
          f"device_count = {torch.cuda.device_count()}, "
          f"current_device = {torch.cuda.current_device()}")
    
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=base_model_path,
        torch_dtype=torch.bfloat16,
        use_cache=False
    )
    # ==== 使用FSDP封装模型 ====
    model = get_model(model, rank)


    # 加载参考模型并用FSDP封装
    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        use_cache=False
    )
    ref_model = get_model(ref_model, rank)  # 使用相同的封装策略
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=base_model_path,
        padding_side="left",
        use_cache=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    
    # ==== 修改4: 设备设置为当前rank ====
    device = torch.device(f"cuda:{rank}")
    
    # ==== 修改5: 分布式训练函数调用 ====
    trained_model = train_with_grpo(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_data=train_data,
        rank=rank,  # 新增参数
        world_size=world_size,  # 新增参数
        **training_config
    )
    
    # ==== 修改6: 只在主进程保存模型 ====
    if rank == 0:
        print("\nFinal model evaluation after GRPO RL fine-tuning:")
        pre_grpo_accuracy = evaluate_model(ref_model, tokenizer, eval_data, device)
        post_grpo_accuracy = evaluate_model(trained_model, tokenizer, eval_data, device)
        print(f"Pre-GRPO Accuracy: {pre_grpo_accuracy:.2f}%")
        print(f"Post-GRPO Accuracy: {post_grpo_accuracy:.2f}%")
        
        print("\nSaving GRPO fine-tuned model...")
        # with FSDP.state_dict_type(
        #     model, 
        #     StateDictType.FULL_STATE_DICT,
        #     FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
        # ):
        #     cpu_state = model.state_dict()
    if rank == 0:
        trained_model.save_pretrained("dlc_FSDP_model")
        tokenizer.save_pretrained("dlc_FSDP_modell")
    
    cleanup()



if __name__ == "__main__":
    all_data = prepare_dataset("train")
    random.shuffle(all_data)
    size_of_eval_data = 30 
    eval_data = all_data[:size_of_eval_data]
    train_data = all_data[size_of_eval_data:]
    print(f"\nNumber of training examples: {len(train_data)}")
    
    # training_config = {
    #     'num_iterations': 1,
    #     'num_steps': 500,
    #     'batch_size': 1,
    #     'num_generations': 14,
    #     'max_completion_length': 400,
    #     'beta': 0.04,
    #     'learning_rate': 5e-6,
    #     'mu': 1,
    #     'epsilon': 0.1
    # }

    # 调试用
    training_config = {
        'num_iterations': 1,
        'num_steps': 2,
        'batch_size': 4,
        'num_generations': 14,
        'max_completion_length': 400,
        'beta': 0.04,
        'learning_rate': 5e-6,
        'mu': 1,
        'epsilon': 0.1
    }
    
    base_model_path = r"./models/Qwen2.5-1.5B-Instruct"

    num_gpus = torch.cuda.device_count()
    print(f"\nInitializing distributed training on {num_gpus} GPUs...")
    # 使用 mp.spawn 调用 main_fn
    mp.spawn(
        main_fn,
        args=(
            num_gpus,         # world_size
            train_data,       # 完整训练数据
            eval_data,        # 评估数据
            base_model_path,  # 模型路径
            training_config   # 训练配置
        ),
        nprocs=num_gpus,
        join=True
    )


