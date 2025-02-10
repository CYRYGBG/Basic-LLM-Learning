#!/bin/bash
# chmod +x run_eval.sh
# ./run_eval.sh

# 定义所有模型的路径数组
model_paths=(
    "D:\04.Code\model\Qwen2.5-1.5B" 
    "D:\04.Code\model\Qwen2.5-1.5B-lora-sft"
)

# 遍历路径并逐个执行
for model_path in "${model_paths[@]}"; do
    lm_eval --model hf --model_args pretrained="$model_path" --tasks gsm8k --device cuda:0 --batch_size 16 --output_path ./eval_results --num_fewshot 0
done

