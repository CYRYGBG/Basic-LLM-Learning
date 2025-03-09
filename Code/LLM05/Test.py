import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import warnings
warnings.filterwarnings("ignore")
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
from collections import deque
import shutil

base_model_path = r"./models/Qwen2.5-1.5B-Instruct"

save_path = r"/home/yeqi3/cyr/code/Basic-LLM-Learning/Code/LLM05/models/dlc_1.5B_model"
save_path_tmp = r'./models/AllDataset'


def set_random_seed(seed: int = 42):
    """
    Set the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use for random number generation.

    Returns:
        None

    Explanation:
        1. Sets seed for Python's built-in random module for basic random operations.
        2. Sets seed for NumPy, ensuring consistent random number generation in array operations.
        3. Sets seed for PyTorch CPU operations.
        4. If CUDA is available, sets seed for all GPU devices.
        5. Configures cuDNN to ensure deterministic behavior:
           - Sets deterministic flag to True, ensuring reproducible results.
           - Disables benchmarking to prevent algorithm selection based on hardware.

    Note:
        Setting deterministic behavior may impact performance but ensures consistent results
        across multiple runs, which is crucial for debugging and research.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call the function to set random seed for reproducibility
set_random_seed(42)

from torch.utils.tensorboard import SummaryWriter



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

# 从模型输出中获取<answer>和</answer>之间的部分
def extract_answer_from_model_output(text):
   """
   Extracts the value from the last <answer> tag in the text.

   Args:
       text (str): The model-generated text containing XML-style <answer> tags.

   Returns:
       str or None: The content inside the <answer> tags, or None if no valid answer is found.

   Explanation:
       1. Splits the text on the <answer> tag to isolate content after the tag.
       2. Checks if at least one <answer> tag exists in the text.
       3. For the last <answer> segment:
          - Verifies it contains a closing </answer> tag.
          - Extracts only the content between the tags.
       4. Returns None if the answer is empty (just "...") or if tags are missing.
   """
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

# 从数据集中提取答案
def extract_answer_from_dataset(text):
   """
   Extracts the answer from the GSM8K dataset examples.

   Args:
       text (str): The dataset example text containing a question and answer.

   Returns:
       str or None: The extracted answer part after the '####' delimiter, or None if not found.

   Explanation:
        1. 检查文本是否包含用于分隔问题和答案的 `####` 分隔符。
        2. 如果找到，则在分隔符处分割文本并返回第二部分（即答案）。
        3. 答案会去除首尾的空白字符。
        4. 如果不存在分隔符，则返回 None。
   """
   if "####" not in text:
       return None
   return text.split("####")[1].strip()




def prepare_dataset(split="train"):
    """
    Load and prepare the GSM8K dataset for training with string prompts.

    Args:
        split (str): The dataset split to load ("train" or "test"). Defaults to "train".

    Returns:
       list: A list of formatted examples, each containing a prompt string and answer.

    Explanation:
        1. 从 Hugging Face 的 datasets hub 加载 GSM8K 数据集。
        2. 对于数据集中的每个示例：
        - 创建一个包含系统提示和问题的消息列表。
        - 使用 `build_prompt()` 将此列表转换为单个字符串提示。
        - 从数据集示例中提取答案。
        - 创建一个包含提示和答案的格式化示例字典。
        3. 返回准备好的格式化示例列表，用于模型训练或评估。
    """
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
   """
   Build a single prompt string from a list of messages.

   Args:
       messages (list): A list of message dictionaries, each with 'role' and 'content' keys.

   Returns:
       str: A concatenated string of all message contents.

   Explanation:
        1. 获取一个以典型聊天格式组织的消息字典列表。
        2. 从每条消息中提取 `content` 字段并去除空白字符。
        3. 将所有内容字符串用换行符连接，生成一个单一的提示。
        4. 这种方法在从结构化消息转换为字符串的同时，保留了训练格式。
   """
   return "\n".join([msg["content"].strip() for msg in messages])




# 从文本中提取最后一个数字
def extract_last_number(text):
   """
   Extracts the last number appearing in the text.

   Args:
       text (str): The text to extract a number from.

   Returns:
       float or None: The last number in the text, or None if no number is found.

   Explanation:
       1. Removes dollar signs and percent symbols from the text.
       2. Uses regex to find a number that appears at the end of the text (possibly after whitespace).
       3. The pattern matches numbers that appear at the end of the string, with or without decimal points.
       4. Returns the found number as a float, or None if no match is found.
   """
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
   """
   Extracts a single number from text if exactly one number is present.

   Args:
       text (str): The text to extract a number from.

   Returns:
       float or None: The single number in the text, or None if zero or multiple numbers are found.

   Explanation:
       1. Uses regex to find all numbers in the text (including negative numbers and decimals).
       2. If exactly one number is found, returns it as a float.
       3. If zero or multiple numbers are found, returns None.
   """
   numbers = re.findall(r'-?\d*\.?\d+', text)
   return float(numbers[0]) if len(numbers) == 1 else None

def evaluate_model(model, tokenizer, eval_examples, device):
   """
   Evaluates the model on a set of examples and prints detailed results.

   Args:
        model: 待评估的语言模型。
        tokenizer: 用于编码输入和解码输出的分词器。
        eval_examples (list): 评估示例列表，每个示例包含 "prompt" 和 "answer"。
        device: 运行评估的设备, CPU 或 GPU。


   Returns:
       float: The accuracy percentage (correct predictions / total examples * 100).

   Explanation:
        1. 将模型设置为评估模式。
        2. 对于评估集中的每个示例：
            - 对提示进行编码并使用模型生成响应。
            - 从生成的响应中提取预测的答案。
            - 使用多种方法将预测的答案与预期的答案进行比较：
                a. 精确字符串匹配
                b. 单数字提取和比较
                c. 最后一个数字提取和比较
            - 打印每个示例的详细信息。
        3. 计算并返回总体准确率。
        4. 将模型恢复为训练模式。

   """
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
   """
   Combines correctness and format rewards.

   Args:
       prompts (list[str]): List of prompt texts
       completions (list[list[dict]]): List of completion dictionaries
       answer (list[str]): List of expected answers

   Returns:
       list[float]: Combined rewards for each prompt-completion pair

   Explanation:
       1. Calculates separate rewards for correctness and format compliance.
       2. Combines the rewards with the following weights:
          - Correctness score range: 0.0 to 2.0
          - Format score range: 0.0 to 1.0
          - Total possible range: 0.0 to 3.0
       3. Returns the combined reward for each example.
   """
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
    """
    计算输入序列末尾指定数量 token 的对数概率。

    参数:
        model: 语言模型（如 HuggingFace 的预训练模型）。
        input_ids (torch.Tensor): 输入序列的 token ID，形状为 [batch_size, seq_len]。
        attention_mask (torch.Tensor): 注意力掩码，标识有效 token 位置，形状与 input_ids 一致。
        logits_to_keep (int): 需要计算对数概率的末尾 token 数量（保留最后几个 token）。

    返回:
        torch.Tensor: 选中 token 的对数概率，形状为 [batch_size, logits_to_keep]。

    实现步骤:
        1. 获取模型输出的 logits，并移除最后一个位置的预测（因预测的是下一个 token）。
        2. 从 input_ids 中截取最后 `logits_to_keep` 个 token 作为目标。
        3. 从 logits 中截取对应的最后 `logits_to_keep` 个位置的预测结果。
        4. 调用 `selective_log_softmax` 计算目标 token 的对数概率。
    """
    # 获取模型输出 logits，并移除最后一个 token 的预测（形状变为 [batch_size, seq_len-1, vocab_size]）
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    
    # 截取 input_ids 的最后 logits_to_keep 个 token（目标 token）
    input_ids = input_ids[:, -logits_to_keep:]
    
    # 截取 logits 的最后 logits_to_keep 个位置（对应目标 token 的预测）
    logits = logits[:, -logits_to_keep:, :]
    
    # 计算目标 token 的对数概率
    return selective_log_softmax(logits, input_ids)



def create_completion_mask(completion_ids, eos_token_id):
    """
    生成一个掩码，用于屏蔽生成序列中首个 EOS token 之后的所有 token。

    参数:
        completion_ids (torch.Tensor): 生成序列的 token ID，形状为 [batch_size, seq_len]。
        eos_token_id (int): 结束符（EOS）的 token ID。

    返回:
        torch.Tensor: 二进制掩码，形状为 [batch_size, seq_len]，有效 token 为 1，EOS 之后为 0。

    实现步骤:
        1. 标记 EOS 出现的位置。
        2. 初始化默认 EOS 索引为序列长度（处理无 EOS 的情况）。
        3. 确定哪些序列中存在 EOS。
        4. 对有 EOS 的序列，更新首个 EOS 的位置索引。
        5. 生成位置索引矩阵，并与 EOS 索引比较生成掩码。
    """
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



def generate_completions(model, tokenizer, prompts, num_generations=4, max_completion_length=32):
    """
    为每个提示(prompt)生成多个补全(completion)。

    Args:
        model: 语言模型对象
        tokenizer: 用于文本编码/解码的分词器
        prompts (list): 文本提示列表
        num_generations (int): 每个提示生成的补全数量
        max_completion_length (int): 生成的最大token数量

    Returns:
        tuple: 包含四个元素的元组，格式为 (prompt_ids, prompt_mask, completion_ids, completion_mask)

    Explanation:
        1. 编码输入提示(prompts)并将其移动到计算设备
        2. 每个提示重复多次以实现批量生成
        3. 使用模型生成补全内容
        4. 从输出中分离生成的补全部分
        5. 创建补全的注意力掩码
    """
    # 选择计算设备（优先使用GPU）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 编码输入提示(prompts)
    inputs = tokenizer(
        prompts, 
        return_tensors="pt",  # 返回PyTorch tensor格式
        padding=True,         # 启用填充
        padding_side="left"   # 左侧填充
    )
    
    # 将输入数据移动到计算设备
    prompt_ids = inputs["input_ids"].to(device)
    prompt_mask = inputs["attention_mask"].to(device)
    
    # 打印调试信息（输入批量大小和设备）
    # print(f"Input batch size: {prompt_ids.size(0)}, Device before model: {prompt_ids.device}")
    
    # 记录提示文本的原始长度（用于后续截取生成的补全）
    prompt_length = prompt_ids.size(1)
    
    # 沿批次维度(batch dimension)重复每个提示
    # 示例：num_generations=4时将批次扩大4倍（每个原始样本生成4个补全）
    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)
    
    # 使用模型生成文本
    outputs = model.generate(
        input_ids=prompt_ids,
        attention_mask=prompt_mask,
        max_new_tokens=max_completion_length,  # 生成的最大新token数
        do_sample=True,         # 启用随机采样
        temperature=1.0,        # 采样温度参数
        pad_token_id=tokenizer.pad_token_id,   # 填充token ID
        eos_token_id=tokenizer.eos_token_id,   # 结束token ID
        early_stopping=False    # 禁用提前停止（生成指定长度）
    )
    
    # 打印生成后的调试信息
    print(f"Output batch size: {outputs.size(0)}, Device after model: {outputs.device}")
    
    # 提取生成的补全部分（截取掉原始的提示内容）
    completion_ids = outputs[:, prompt_length:]
    
    # 创建补全的注意力掩码（用于后续处理）
    completion_mask = create_completion_mask(completion_ids, tokenizer.eos_token_id)
    
    return prompt_ids, prompt_mask, completion_ids, completion_mask



def generate_rollout_data(model, ref_model, tokenizer, batch_samples, num_generations, max_completion_length):
    """
    生成用于GRPO训练的交互数据，包括生成文本及其对数概率。

    Args:
        model: 当前训练的强化学习策略模型
        ref_model: 参考模型（用于KL散度计算的基准模型）
        tokenizer: 文本编码/解码器
        batch_samples (list): 训练样本组成的批次（每个样本包含prompt和answer）
        num_generations (int): 每个样本生成的补全数量
        max_completion_length (int): 生成文本的最大token长度

    Returns:
        dict: 包含GRPO训练所需完整数据的字典

    Explanation:
        1. 解析输入批次中的提示和标准答案
        2. 使用当前模型生成多条补全文本
        3. 拼接提示与生成文本得到完整序列
        4. 计算策略模型和参考模型的生成文本对数概率
        5. 格式化生成文本以计算奖励
        6. 对齐输入数据维度（根据生成数量扩展样本）
    """
    # 检测计算设备（优先使用GPU）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 从训练样本中提取提示和答案
    # 支持字典和元组两种数据格式
    # 示例：
    # 'prompt': 'Respond in the following format:\n<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\nNatalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?'
    # 'answer': '72'
    prompts = [sample["prompt"] if isinstance(sample, dict) else sample[0] for sample in batch_samples]
    answers = [sample["answer"] if isinstance(sample, dict) else sample[1] for sample in batch_samples]

    # 禁用梯度计算（生成阶段）
    with torch.no_grad():
        # 生成多个补全文本 (shape: (batch_size * num_generations, ...))"
        prompt_ids, prompt_mask, completion_ids, completion_mask = generate_completions(
            model, tokenizer, prompts, num_generations, max_completion_length
        )
        
        # 拼接提示和补全作为完整输入序列
        # 维度: (batch_size*num_generations, prompt_len+completion_len)
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)             
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)     # 相同的维度结构
        
        # 记录需要保留的logits数量（补全文本的token长度）
        logits_to_keep = completion_ids.size(1)
        
        # 计算模型对生成内容的对数概率
        old_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)       # 当前策略模型
        ref_log_probs = compute_log_probs(ref_model, input_ids, attention_mask, logits_to_keep)   # 参考模型

    # 解码生成文本用于后续奖励计算
    formatted_completions = [[
        {'content': tokenizer.decode(ids, skip_special_tokens=True)}  # 解码并跳过特殊token
    ] for ids in completion_ids]  # 每个生成样本包装成字典列表形式

    # 扩展原始样本以匹配生成的补全数量
    repeated_prompts = [p for p in prompts for _ in range(num_generations)]    # 格式示例：[prompt1, prompt1, prompt1..., prompt2, prompt2...]
    repeated_answers = [a for a in answers for _ in range(num_generations)]   # 同理扩展答案

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


eval_data = prepare_dataset("test")[:50]
print()
print(f"Number of testing examples: {len(eval_data)}")


def optimize_model_memory(model):
    """
    Optimizes the model to use less memory during training.

    Args:
        model: The language model to optimize.

    Returns:
        The optimized model.

    Explanation:
        1. Sets the model to training mode.
        2. Disables KV caching to save memory.
        3. Enables gradient checkpointing to trade computation for memory.
        4. Ensures that input embeddings require gradients:
           - Either uses the built-in method if available.
           - Or adds a forward hook to the input embeddings layer.
        5. Returns the optimized model ready for memory-efficient training.
    """
    model.train()
    model.config.use_cache = False

    # First ensure inputs will require gradients
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Then enable gradient checkpointing
    model.gradient_checkpointing_enable()

    return model

# Main execution
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using primary device: {device}")

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
output_dir = "math_solver_model"


# 加载原始模型
original_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_cache=False  
)
original_model = optimize_model_memory(original_model)
original_model.eval()  
original_tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=base_model_path,
    padding_side="left"
)
original_tokenizer.pad_token = original_tokenizer.eos_token

# 加载微调后的模型
print("\nLoading fine-tuned model for evaluation...")
finetuned_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=save_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_cache=False  
)
finetuned_model = optimize_model_memory(finetuned_model)
finetuned_model.eval()  
finetuned_tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=save_path,
    padding_side="left"
)
finetuned_tokenizer.pad_token = finetuned_tokenizer.eos_token


# 加载更多步数的模型
print("\nLoading fine-tuned model for evaluation...")
finetuned_model_tmp = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=save_path_tmp,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_cache=False  
)
finetuned_model_tmp = optimize_model_memory(finetuned_model_tmp)
finetuned_model_tmp.eval()  
finetuned_tokenizer_tmp = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=save_path_tmp,
    padding_side="left"
)
finetuned_tokenizer_tmp.pad_token = finetuned_tokenizer.eos_token


# 评估原始模型
print("\nEvaluating original model...")
with torch.no_grad():
    original_accuracy = evaluate_model(
        model=original_model,
        tokenizer=original_tokenizer,
        eval_examples=eval_data,
        device=device
    )

# 评估微调后的模型
print("\nEvaluating fine-tuned model...")
with torch.no_grad():
    finetuned_accuracy = evaluate_model(
        model=finetuned_model,
        tokenizer=finetuned_tokenizer,  
        eval_examples=eval_data,
        device=device
    )

# 评估更多步数的模型
print("\nEvaluating fine-tuned model...")
with torch.no_grad():
    finetuned_accuracy_tmp = evaluate_model(
        model=finetuned_model_tmp,
        tokenizer=finetuned_tokenizer_tmp,  
        eval_examples=eval_data,
        device=device
    )

print(f"Original Model Accuracy: {original_accuracy:.2f}%")
print(f"Fine-tuned Model Accuracy: {finetuned_accuracy:.2f}%")
print(f"More-tuned Model Accuracy: {finetuned_accuracy_tmp:.2f}%")

