#!/bin/bash

# 测试参数配置
batch_sizes=(1 8 16)
seq_lengths=(128 256 512 1024 2048 4096)

# batch_sizes=(1 8)
# seq_lengths=(128 256)


# 输出文件配置（自动添加时间戳）
output_dir="benchmark_results"
timestamp=$(date +"%m%d_time%H%M")
output_file="$output_dir/date${timestamp}_results.txt"

# 检查并创建输出目录
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
    echo "Created output directory: $output_dir"
fi

# 清空或创建输出文件
: > "$output_file"

# 记录开始时间
start_time=$(date +%s)
echo "Benchmark start at: $(date)"

# 主测试循环
for bs in "${batch_sizes[@]}"; do
    echo "=== Testing batch_size: $bs ===" | tee -a "$output_file"
    
    for seq in "${seq_lengths[@]}"; do
        echo "Running test: bs=$bs, seq=$seq"
        
        # 执行测试并捕获输出
        result=$(python benchmark2.py --batch_size $bs --seq_len $seq 2>&1)
        
        # 将结果写入文件
        {
            echo "bs:$bs | h_num:12 | seq:$seq"
            echo "$result"
            echo " "
        } >> "$output_file"
    done
done

# 计算总耗时
end_time=$(date +%s)
duration=$((end_time - start_time))

# 完成信息
{
    echo ""
    echo "Benchmark end at: $(date)"
    echo "Total test duration: ${duration} seconds"
} >> "$output_file"

echo "All tests completed. Results saved to: $output_file"
echo "Total time: ${duration}s"