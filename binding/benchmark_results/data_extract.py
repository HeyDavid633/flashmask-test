import re

# 输入和输出文件名
input_file = 'date0713_time1714_results.txt'
output_file = 'FlashAttn2_full_baseline.txt'

# 正则表达式模式，用于匹配Bind Kernel行并提取相关信息
pattern = re.compile(r'bs:(\d+)\s+\|\s+seq:(\d+)\s+\|\s+Bind Kernel\s+:\s+([\d.]+)\s+ms\s+/')

# 读取输入文件并提取数据
with open(input_file, 'r') as f:
    content = f.read()

# 查找所有匹配项
matches = pattern.findall(content)

# 将提取的数据写入输出文件，格式化为FlashAttn2的格式
with open(output_file, 'w') as f:
    for match in matches:
        bs, seq, time = match
        # 格式化输出行
        output_line = f"bs:{bs} | seq:{seq} | FlashAttn2: {time} ms/iter\n"
        f.write(output_line)

print(f"处理完成，结果已保存到 {output_file}")