#!/bin/bash

# 测试脚本: run_benchmark.sh

# 配置测试参数
BATCH_SIZES=(1 8 16)
SEQ_LENS=(128 256 512 1024 2048 4096)
# SEQ_LENS=(128)

# 输出文件
OUTPUT_FILE="splat_benchmark_results_A100_fp16.txt"
EXECUTABLE="./splat_my_test_fp16_exec"  # 修改为您的可执行文件路径

# 检查可执行文件是否存在
if [ ! -f "$EXECUTABLE" ]; then
    echo "错误: 可执行文件 $EXECUTABLE 不存在"
    echo "请先编译程序: make"
    exit 1
fi

# 创建或清空输出文件
echo "=== SPLAT FlashAttention 性能测试结果 ===" > $OUTPUT_FILE
echo "测试时间: $(date)" >> $OUTPUT_FILE
echo "==========================================" >> $OUTPUT_FILE
echo "Batch Size | Seq Length | Time (ms/iter)" >> $OUTPUT_FILE
echo "-----------|------------|----------------" >> $OUTPUT_FILE

# 计数器
total_tests=$(( ${#BATCH_SIZES[@]} * ${#SEQ_LENS[@]} ))
current_test=0

echo "开始性能测试..."
echo "总共需要测试: $total_tests 个配置组合"
echo ""

# 遍历所有配置组合
for batch in "${BATCH_SIZES[@]}"; do
    for seq_len in "${SEQ_LENS[@]}"; do
        current_test=$((current_test + 1))
        echo "[$current_test/$total_tests] 测试 batch_size=$batch, seq_len=$seq_len ..."
        
        output=$($EXECUTABLE $batch $seq_len 2>&1)
        
        echo "$output" >> $OUTPUT_FILE

    done
done
