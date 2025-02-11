#!/bin/bash

#============= 参数检查 =============#
if [[ $# -lt 1 ]] || [[ "$1" == "-h" ]]; then
    echo "多GPU训练监控脚本"
    echo "用法：$0 [GPU设备列表] [其他参数传递给训练命令]"
    echo "示例："
    echo "  $0 0,1          监控GPU 0和1"
    echo "  $0 0-3          监控GPU 0到3"
    echo "  $0 0,2-4 --fp16 复杂设备组合并传递参数"
    exit 1
fi

#============= 关键函数恢复 =============#
validate_devices() {
    local max_gpu=$(nvidia-smi -L | wc -l)
    for dev in $(echo "$ACTIVE_DEVICES" | tr ',' ' '); do
        if [[ ! "$dev" =~ ^[0-9]+$ ]] || (( dev >= max_gpu )); then
            echo "[错误] 无效GPU设备ID: $dev (最大可用ID: $((max_gpu-1)))" >&2
            exit 2
        fi
    done
}

#============= 路径修复 =============#
# 需修改的代码段：
REPORT_DIR="./gpu_report"      # ← 改成你想要的绝对路径
mkdir -p "$REPORT_DIR" || {
    echo "[错误] 无法创建报告目录: $REPORT_DIR" >&2
    exit 1
}
REPORT_FILE="${REPORT_DIR}/$(date +%Y%m%d_%H%M%S)_gpu_report.md"    
TEMP_DATA=$(mktemp)
PEAK_MEM_FILE=$(mktemp)

#============= 设备处理逻辑 =============#
parse_devices() {
    echo "$DEVICES" | sed 's/-/ /g' | awk '
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

#============= 主逻辑修复 =============#
DEVICES="$1"
shift  # 正确处理剩余参数
ACTIVE_DEVICES=$(parse_devices "$DEVICES")
validate_devices  # 调用验证函数

#============= 监控进程修复 =============#
monitor_gpu() {
    local peak=0
    while sleep 0.5; do
        raw_data=$(nvidia-smi -i $ACTIVE_DEVICES \
            --query-gpu=index,memory.used,memory.total \
            --format=csv,noheader,nounits 2>/dev/null)
        
        current_sum=$(echo "$raw_data" | awk -F', ' '
        BEGIN { sum = 0 }
        { 
            gsub(/[^0-9]/, "", $2);  # 清洗数据
            sum += $2 
        } 
        END { print sum }')
        
        if [[ $current_sum -gt $peak ]]; then
            peak=$current_sum
            echo $peak > $PEAK_MEM_FILE
        fi
        
        echo "$raw_data" > "$TEMP_DATA"  # 保留详细数据
    done
}

#============= 执行逻辑修正 =============#
monitor_gpu &
monitor_pid=$!
sleep 1  # 等待监控启动

echo "指定设备: GPU $ACTIVE_DEVICES" | tee -a "$REPORT_FILE"  # 修复tee路径

start_time=$(date +%s)
CUDA_VISIBLE_DEVICES=$ACTIVE_DEVICES $@  # 修复命令执行方式
exit_code=$?
end_time=$(date +%s)

#============= 结果收集 =============#
kill $monitor_pid 2>/dev/null
runtime=$((end_time - start_time))
peak_total=$(cat $PEAK_MEM_FILE || echo 0)

#============= 报告生成修复 =============#
{
    echo "# 训练报告"
    echo "**执行命令**: \`CUDA_VISIBLE_DEVICES=$ACTIVE_DEVICES $@\`"
    echo ""
    echo "## 综合统计"
    echo "| 运行时间 | 峰值显存总和 |"
    echo "|----------|--------------|"
    printf "| %02d:%02d:%02d | %6d MB |\n" \
        $((runtime/3600)) $((runtime%3600/60)) $((runtime%60)) $peak_total
    
    echo ""
    echo "## 设备详情"
    echo "| 设备ID | 显存使用 | 总显存 | 使用率 |"
    echo "|--------|----------|--------|--------|"
    while IFS=', ' read -r idx use total; do
        (( total == 0 )) && continue  # 防止除零错误
        pct=$(awk "BEGIN {printf \"%.1f\", $use*100/$total}")%
        printf "| %6d | %5d MB | %5d MB | %7s |\n" $idx $use $total "$pct"
    done < "$TEMP_DATA"
} > "$REPORT_FILE"

# 清理资源
rm -f "$TEMP_DATA" "$PEAK_MEM_FILE"
echo "报告生成完成: $REPORT_FILE"
echo -e "\n训练报告路径: $(realpath "$REPORT_FILE")"
