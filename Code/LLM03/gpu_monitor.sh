#!/bin/bash
# 文件名：gpu_monitor
# 使用方式：直接运行 ./gpu_monitor

#################### 用户配置区 ####################
DEVICES="0"                     # 需要使用的GPU设备ID（支持,和-格式）
declare -a ENV_VARS=(             # 需要设置的环境变量（数组格式）
    # "FORCE_TORCHRUN=1"
)
declare -a TRAIN_COMMAND=(        # 要执行的训练命令（数组格式）
    "llamafactory-cli"
    "train"
    "D:\\CYR_TMP\\Basic-LLM-Learning\\Code\\LLM03\\train_yaml\\loraplus16.yaml"
)
##################################################

################### 初始化设置 ####################
REPORT_DIR="D:\CYR_TMP\Basic-LLM-Learning\Code\LLM03\gpu_report"  # 报告存储目录
REPORT_FILE="${REPORT_DIR}/$(date +%Y%m%d_%H%M%S)_gpu_report.md"
mkdir -p "$REPORT_DIR" || {
    echo "[错误] 无法创建报告目录: $REPORT_DIR" >&2
    exit 1
}

PEAK_PER_GPU_FILE=$(mktemp)
TEMP_DATA=$(mktemp)

#################### 基础函数 ######################
parse_devices() {
    echo "$1" | sed 's/-/ /g' | awk '
    BEGIN { FS=","; OFS="," }
    {
        for(i=1; i<=NF; i++) {
            if($i ~ /[0-9]+ [0-9]+/) {
                split($i, range, " ")
                for(j=range[1]; j<=range[2]; j++) 
                    printf "%s%s", j, (j==range[2] && i==NF ? "" : ",")
            } else {
                printf "%s%s", $i, (i==NF ? "" : ",")
            }
        }
    }'
}

validate_devices() {
    local max_gpu=$(nvidia-smi -L | wc -l)
    for dev in $(echo "$ACTIVE_DEVICES" | tr ',' ' '); do
        if [[ ! "$dev" =~ ^[0-9]+$ ]] || (( dev >= max_gpu )); then
            echo "[错误] 无效GPU设备ID: $dev (最大可用ID: $((max_gpu-1)))" >&2
            exit 2
        fi
    done
}

monitor_gpu() {
    local devices
    IFS=',' read -ra devices <<< "$ACTIVE_DEVICES"
    declare -A peak_mem=()
    
    # 初始化显存记录文件
    for dev in "${devices[@]}"; do
        peak_mem[$dev]=0
        > "${MEM_LOG_DIR}/gpu_${dev}.log"
    done

    while sleep 0.5; do
        raw_data=$(nvidia-smi -i $ACTIVE_DEVICES \
            --query-gpu=index,memory.used \
            --format=csv,noheader,nounits 2>/dev/null)

        while IFS=', ' read -r index mem_used; do
            index=${index//[^0-9]/}
            mem_used=${mem_used//[^0-9]/}
            [[ -z "$mem_used" ]] && continue

            # 更新峰值显存
            if [[ ${peak_mem[$index]} -lt $mem_used ]]; then
                peak_mem[$index]=$mem_used
            fi
            
            # 记录显存使用情况
            echo "$mem_used" >> "${MEM_LOG_DIR}/gpu_${index}.log"

        done <<< "$raw_data"

        # 更新峰值记录文件
        : > "$PEAK_PER_GPU_FILE"
        for idx in "${!peak_mem[@]}"; do
            echo "$idx ${peak_mem[$idx]}" >> "$PEAK_PER_GPU_FILE"
        done
    done
}

################### 主执行逻辑 ###################
# 处理设备参数
ACTIVE_DEVICES=$(parse_devices "$DEVICES")
validate_devices

# 创建临时目录存放显存日志
MEM_LOG_DIR=$(mktemp -d)
export MEM_LOG_DIR

# 启动显存监控
monitor_gpu &
monitor_pid=$!
sleep 1

# 执行训练任务
start_time=$(date +%s)
echo "====== 开始训练 ======="
echo "使用设备: GPU $ACTIVE_DEVICES"
echo "执行命令: ${TRAIN_COMMAND[@]}"

# 执行命令
for var in "${ENV_VARS[@]}"; do
    export "$var"
done
"${TRAIN_COMMAND[@]}"
exit_code=$?
end_time=$(date +%s)

# 清理监控进程
kill $monitor_pid 2>/dev/null
runtime=$((end_time - start_time))

# 读取显存数据
declare -A peak_mem=()
if [[ -f "$PEAK_PER_GPU_FILE" ]]; then
    while read -r idx mem; do
        peak_mem["$idx"]=$mem
    done < "$PEAK_PER_GPU_FILE"
fi

# 计算显存统计
peak_total=0
num_gpus=${#peak_mem[@]}
for mem in "${peak_mem[@]}"; do
    peak_total=$((peak_total + mem))
done
average_peak=$((num_gpus > 0 ? peak_total / num_gpus : 0))

# 计算平均显存使用 (修正点：变量名错误修复)
declare -A avg_mem=()
total_avg=0
total_count=0

for idx in "${!peak_mem[@]}"; do
    log_file="${MEM_LOG_DIR}/gpu_${idx}.log"
    if [[ -s "$log_file" ]]; then
        sum=0
        count=0
        while read -r value; do
            sum=$((sum + value))
            ((count++))
        done < "$log_file"
        # 修正变量名错误：aavg -> avg
        avg=$(awk -v sum="$sum" -v count="$count" 'BEGIN{ printf "%.2f", (sum / count) }')
        avg_mem["$idx"]=$avg
        total_avg=$(awk -v prev="$total_avg" -v curr="$avg" 'BEGIN{ printf "%.2f", (prev + curr) }')
        ((total_count++))
    else
        avg_mem["$idx"]=0
    fi
done

# 计算总平均值
overall_avg=0
if [[ $total_count -gt 0 ]]; then
    overall_avg=$(awk -v total_avg="$total_avg" -v total_count="$total_count" '
                    BEGIN {
                        avg = (total_count != 0) ? total_avg / total_count : 0.00
                        printf "%.2f\n", avg
                    }')
fi

################ 生成报告 ################
{
    echo "# 训练执行报告"
    echo "执行命令: \`${TRAIN_COMMAND[@]}\`"

    echo "## 基础信息"
    echo "| 项目        | 值                           |"
    echo "|-------------|------------------------------|"
    echo "| 执行时间    | $(date -d "@$start_time" '+%Y-%m-%d %H:%M:%S') |"
    echo "| 使用设备    | GPU $ACTIVE_DEVICES          |"
    echo "| 退出状态码  | $exit_code                   |"

    echo -e "\n## 资源消耗"
    echo "### 总体统计"
    echo "| 项目             | 值                 |"
    echo "|------------------|--------------------|"
    printf "| 运行时间         | %02d:%02d:%02d           |\\n" $((runtime/3600)) $((runtime%3600/60)) $((runtime%60))
    printf "| 峰值显存总和     | %'d MB          |\\n" "$peak_total"
    printf "| 平均峰值显存     | %'d MB          |\\n" "$average_peak"
    printf "| 总平均显存       | %.2f MB          |\\n" "$overall_avg"
    
    echo -e "\n### 各GPU显存使用详情"
    echo "| GPU ID | 峰值显存 (MB) | 平均显存 (MB) |"
    echo "|--------|---------------|---------------|"
    for idx in "${!peak_mem[@]}"; do
        printf "| %-6d | %'13d | %'13.2f |\\n" "$idx" "${peak_mem[$idx]}" "${avg_mem[$idx]}"
    done

} >| "$REPORT_FILE"

# 清理临时文件
rm -rf "$MEM_LOG_DIR"
rm -f "$PEAK_PER_GPU_FILE" "$TEMP_DATA"

echo -e "\n✅ 训练完成"
echo "报告路径: $(realpath "$REPORT_FILE")"
