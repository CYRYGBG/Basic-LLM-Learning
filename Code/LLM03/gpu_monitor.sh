#!/bin/bash
# 文件名：gpu_monitor
# 使用方式：直接运行 ./gpu_monitor

#################### 用户配置区 ####################
DEVICES="0,1"  
# 需要设置的环境变量（数组格式）
declare -a ENV_VARS=(
    "FORCE_TORCHRUN=1"
)                  # 要使用的GPU设备ID（支持,和-格式）
declare -a TRAIN_COMMAND=(       # 要执行的训练命令（数组格式） 
    "llamafactory-cli"
    "train"
    "/root/Basic-LLM-Learning/Code/LLM03/train_yaml/full.yaml"
)
##################################################

################### 初始化设置 ####################
REPORT_DIR="/root/Basic-LLM-Learning/Code/LLM03/gpu_report"  # 报告存储目录
REPORT_FILE="${REPORT_DIR}/$(date +%Y%m%d_%H%M%S)_gpu_report.md"
mkdir -p "$REPORT_DIR" || {
    echo "[错误] 无法创建报告目录: $REPORT_DIR" >&2
    exit 1
}

PEAK_MEM_FILE=$(mktemp)
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
    local peak=0
    while sleep 0.5; do
        raw_data=$(nvidia-smi -i $ACTIVE_DEVICES \
            --query-gpu=index,memory.used,memory.total \
            --format=csv,noheader,nounits 2>/dev/null)
        
        current_sum=$(awk -F', ' '
        BEGIN { sum = 0 }
        { 
            gsub(/[^0-9]/, "", $2);
            sum += $2 
        } 
        END { print sum }' <<< "$raw_data")
        
        if (( current_sum > peak )); then
            peak=$current_sum
            echo $peak | tee "$PEAK_MEM_FILE" >/dev/null
        fi
        
        echo "$raw_data" > "$TEMP_DATA"
    done
}

################### 主执行逻辑 ###################
# 处理设备参数
ACTIVE_DEVICES=$(parse_devices "$DEVICES")
validate_devices

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
# export CUDA_VISIBLE_DEVICES="$ACTIVE_DEVICES"
for var in "${ENV_VARS[@]}"; do
    export "$var"
done
"${TRAIN_COMMAND[@]}"
exit_code=$?
end_time=$(date +%s)

# 清理监控进程
kill $monitor_pid 2>/dev/null
runtime=$((end_time - start_time))
peak_total=$(cat "$PEAK_MEM_FILE" 2>/dev/null || echo 0)

################ 生成报告 ################
{
    echo "# 训练执行报告"
    echo "${TRAIN_COMMAND[@]}"

    echo "## 基础信息"
    echo "| 项目        | 值                           |"
    echo "|-------------|------------------------------|"
    echo "| 执行时间    | $(date -d "@$start_time" '+%Y-%m-%d %H:%M:%S') |"
    echo "| 使用设备    | GPU $ACTIVE_DEVICES          |"
    echo "| 退出状态码  | $exit_code                   |"

    echo -e "\n## 资源消耗"
    echo "| 项目         | 值              |"
    echo "|--------------|-----------------|"
    printf "| 运行时间     | %02d:%02d:%02d     |\\n" $((runtime/3600)) $((runtime%3600/60)) $((runtime%60))
    printf "| 峰值显存总和 | %'d MB       |\\n" "$peak_total"

    # echo -e "\n## 设备详情"
    # echo "| 设备ID | 显存使用    | 总显存      | 使用率  |"
    # echo "|--------|------------|------------|---------|"
    # while IFS=', ' read -r idx use total; do
    #     (( total == 0 )) && continue
    #     if [[ "$use" =~ ^[0-9]+$ ]]; then
    #         pct=$(awk "BEGIN {printf \"%.1f\", $use*100/$total}")
    #         printf "| %6d | %'8d MB | %'8d MB | %6.1f%% |\\n" "$idx" "$use" "$total" "$pct"
    #     else
    #         printf "| %6d | %8s MB | %'8d MB | %6s |\\n" "$idx" "N/A" "$total" "N/A"
    #     fi
    # done < "$TEMP_DATA"

} >| "$REPORT_FILE"

# 清理临时文件
rm -f "$PEAK_MEM_FILE" "$TEMP_DATA"

echo -e "\n✅ 训练完成"
echo "报告路径: $(realpath "$REPORT_FILE")"
