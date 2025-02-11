#!/bin/bash
# ./monitor.sh 0,1          # 指定GPU 0和1
# ./monitor.sh 1-3          # 指定GPU 1到3
# ./monitor.sh 0,2-4        # 混合指定
# TODO：测试显存计算脚本


# 参数检查与帮助信息
if [[ $# -lt 1 ]] || [[ "$1" == "-h" ]]; then
    echo "多GPU训练监控脚本"
    echo "用法：$0 [GPU设备列表] [其他参数传递给训练命令]"
    echo "示例："
    echo "  $0 0,1 - 监控GPU 0和1"
    echo "  $0 0-2 - 监控GPU 0到2"
    exit 1
fi

# 初始配置
DEVICES="$1"
shift  # 将剩余参数传递给训练命令
REPORT_FILE="$(date +%Y%m%d_%H%M%S)_gpu_report.md"
TEMP_DATA=$(mktemp)

# 设备ID扩展函数：处理0,1-2格式
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

# 验证设备有效性
validate_devices() {
    local max_gpu=$(nvidia-smi -L | wc -l)
    for dev in $(echo "$ACTIVE_DEVICES" | tr ',' ' '); do
        if [[ ! "$dev" =~ ^[0-9]+$ ]] || (( dev >= max_gpu )); then
            echo "[错误] 无效GPU设备ID: $dev (最大可用ID: $((max_gpu-1)))"
            exit 2
        fi
    done
}

# Markdown表格构建器
md_table() {
    local cols=("$@")
    local head="${cols[0]}"
    echo "| ${head//,/ | } |"
    echo "| $(echo "$head" | sed 's/[^,]/---/g' | tr ',' '|') |"
    for line in "${cols[@]:1}"; do
        echo "| ${line//,/ | } |"
    done
}

#----------------- 主程序逻辑 -----------------#
ACTIVE_DEVICES=$(parse_devices "$DEVICES")
validate_devices

# 初始化监控
peak_total=0
gpu_count=$(echo "$ACTIVE_DEVICES" | tr ',' '\n' | wc -l)

# 后台监控进程
(
    while sleep 0.5; do
        usage=$(nvidia-smi -i $ACTIVE_DEVICES \
            --query-gpu=index,memory.used,memory.total \
            --format=csv,noheader,nounits 2>/dev/null | tee "$TEMP_DATA")
        
        current_sum=$(echo "$usage" | awk -F', ' '{sum+=$2} END {print sum}')
        [[ $current_sum -gt $peak_total ]] && peak_total=$current_sum

        peak_ind=$(echo "$usage" | awk -F', ' '{
            mem_pct[$1]=$2/$3*100; 
            if ($2 > peak[$1]) peak[$1]=$2
        } END {
            for(i in peak) printf "%d(%.1f%%),", peak[i], mem_pct[i]
        }' | sed 's/,$//')
    done
) &
monitor_pid=$!

# 执行训练命令
echo "指定设备: GPU $ACTIVE_DEVICES" | tee -a "$REPORT_FILE"
start_time=$(date +%s)
CUDA_VISIBLE_DEVICES=$ACTIVE_DEVICES llamafactory-cli train llama_train.yaml "$@"
exit_code=$?
end_time=$(date +%s)

# 停止监控并收集数据
kill $monitor_pid 2>/dev/null
runtime=$((end_time - start_time))

# 生成Markdown报告
{
    echo "# 多GPU训练监控报告"
    echo "**执行日期**: $(date -d @$start_time '+%Y-%m-%d %H:%M:%S')"
    echo "**执行命令**: \`CUDA_VISIBLE_DEVICES=$ACTIVE_DEVICES llamafactory-cli train llama_train.yaml $@\`"
    echo ""
    echo "## 综合统计"
    md_table "运行时间,峰值显存总量,平均显存占比" \
        "$(printf "%02d:%02d:%02d,%d MB,%.1f%%" \
            $((runtime/3600)) $((runtime%3600/60)) $((runtime%60)) \
            $peak_total \
            $(echo "$peak_ind" | tr ',' '\n' | awk -F'[%(]' '{sum+=$2} END {print sum/NR}'))"

    echo ""
    echo "## 设备详细信息"
    md_table_header="设备ID,总显存(MB),峰值使用量,使用占比"
    md_table_rows=()
    while IFS=', ' read -r idx used total; do
        pct=$(awk "BEGIN {printf \"%.1f\", $used/$total*100}")
        md_table_rows+=("$idx,$total MB,${used} MB (${pct}%)")
    done < "$TEMP_DATA"
    md_table "$md_table_header" "${md_table_rows[@]}"
    
} > "$REPORT_FILE"

# 清理并显示结果
rm "$TEMP_DATA"
echo -e "\n训练报告路径: $(realpath "$REPORT_FILE")"
