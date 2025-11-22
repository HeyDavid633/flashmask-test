import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import re
import matplotlib as mpl
import argparse


def parse_file2(file_path, is_mcfuser=False):
    data = {}
    current_model = None

    with open(file_path, 'r') as f:
        for line in f:
            model_match = re.search(r'e2e (\w+)', line)
            if model_match:
                current_model = model_match.group(1)

            if is_mcfuser:
                submodel_match = re.search(r'$$\w+$$', line)
                if submodel_match:
                    continue

            pattern = r'.*?bs:(\d+).*?seq:(\d+).*?\|\s+([\w\s]+)\s*:\s*([\d.]+)'
            match = re.search(pattern, line)
            if match and current_model:
                bs = int(match.group(1))
                seq = int(match.group(2))
                method = match.group(3).strip()
                time = float(match.group(4))

                if is_mcfuser:
                    method = "MCFuser"

                data.setdefault(bs, {}).setdefault(seq, {}).setdefault(current_model, {})[method] = time

    return data


parser = argparse.ArgumentParser(description='Plot ablation_study')
parser.add_argument('--file_path1', default="ablation-base.txt", required=False)
args = parser.parse_args()
data1 = parse_file2(args.file_path1)
print(data1)

def combinedData(data1):
    combined_data = {}
    target_pairs = [
        (1, 128),
        (8, 512),
        (16, 2048),
    ]

    methods_mapping = {
        'Torch Native': ('data1', 'Torch Native'),
        'STOF Compiled': ('data1', 'STOF Compiled'),
        'STOF MHA': ('data1', 'STOF MHA'),
        'STOF': ('data1', 'STOF'),
    }

    for bs in [1, 8, 16]:
        combined_data[bs] = {}
        target_seqs = [seq for b, seq in target_pairs if b == bs]

        for seq in target_seqs:
            combined_data[bs][seq] = {}

            for model in ['bert_base', 'bert_large', 'gpt', 'llama', 't5', 'Vit_base']:
                combined_data[bs][seq][model] = {}

                for method, (source, key) in methods_mapping.items():
                    if source == 'data1':
                        src_data = data1

                    val = src_data.get(bs, {}).get(seq, {}).get(model, {}).get(key, 0)
                    combined_data[bs][seq][model][method] = val
    return combined_data


combined_data1 = combinedData(data1)


def normalizedData(combined_data):
    normalized_data = {}
    for bs in combined_data:
        normalized_data[bs] = {}
        for seq in combined_data[bs]:
            normalized_data[bs][seq] = {}
            for model in combined_data[bs][seq]:
                model_data = combined_data[bs][seq][model]
                tn_time = model_data.get('Torch Native', 0)

                normalized = {}
                for method, value in model_data.items():
                    if tn_time > 0 and value > 0:
                        normalized[method] = tn_time / value
                    else:
                        normalized[method] = 0

                if tn_time > 0:
                    normalized['Torch Native'] = 1.0

                normalized_data[bs][seq][model] = normalized
    return normalized_data


normalized_data1 = normalizedData(combined_data1)

# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 25

MODELS = ['bert_base', 'bert_large', 'gpt', 'llama', 't5', 'Vit_base']
MODEL_NAMES = ['BERT-Base', 'BERT-Large', 'GPT-2', 'LLaMA', 'T5', 'ViT']
BS_SEQ_PAIRS = [(1, 128), (8, 512), (16, 2048)]
METHODS = ['Torch Native',  'STOF MHA','STOF Compiled', 'STOF']

COLORS = ['#999696','#dff1d7',  '#afc8ea', '#d06c5a']
LABEL_NAMES = ['PyTorch Native', 'Only Operator Fusion', 'Only Unified MHA', 'Unified MHA+Operator Fusion']
num=[0,2,1,3]

def extract_data(normalized_data):
    return {
        (bs, seq): {model: [normalized_data[bs][seq][model][method] for method in METHODS]
                    for model in MODELS}
        for bs, seq in BS_SEQ_PAIRS
    }


data1_extracted = extract_data(normalized_data1)


def plot_subplot(ax, data):
    n_models = len(MODELS)
    n_methods = len(METHODS)
    bar_width = 2
    model_spacing = 2
    group_spacing = 5

    # 计算每组的总宽度
    group_width = n_models * n_methods * bar_width + (n_models - 1) * model_spacing

    # 绘制柱状图
    for pair_idx, (bs, seq) in enumerate(BS_SEQ_PAIRS):
        group_start = pair_idx * (group_width + group_spacing)

        for model_idx, model in enumerate(MODELS):
            model_start = group_start + model_idx * (n_methods * bar_width + model_spacing)

            values = data[(bs, seq)][model]

            for method_idx, value in enumerate(values):
                x = model_start + method_idx * bar_width

                rect = ax.bar(x, value, width=bar_width,
                              color=COLORS[num[method_idx]],
                              edgecolor='black',
                              linewidth=0.8, zorder=3)

                if (value > 4 or value == 4):
                    print(method_idx)
                    if(method_idx==1):
                        ax.text(x-3.3, 3.9, f"{value:.1f}",
                            ha='center', va='top', fontsize=20, color='black')
                    if(method_idx==3):
                        ax.text(x+3.3, 3.9, f"{value:.1f}",
                                ha='center', va='top', fontsize=20, color='black')


    # 添加组间分隔线
    for i in range(1, len(BS_SEQ_PAIRS)):
        x = i * (group_width + group_spacing) - group_spacing / 2 - 1
        ax.axvline(x, color='gray', linestyle='--', linewidth=3)

    # 设置Y轴
    ax.set_ylabel('Speedup', fontsize=36)
    ax.set_ylim(0, 4)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.yaxis.grid(True, linestyle=(0, (10, 5)), linewidth=1)
    ax.margins(x=0.01)

    # 设置X轴标签
    ax.set_xticks([])

    # 添加模型名称标签
    for pair_idx, (bs, seq) in enumerate(BS_SEQ_PAIRS):
        group_start = pair_idx * (group_width + group_spacing)

        for model_idx, model_name in enumerate(MODEL_NAMES):
            model_start = group_start + model_idx * (n_methods * bar_width + model_spacing)
            x = model_start + (n_methods * bar_width) / 2

            # 根据模型名称调整标签位置
            if model_name in ['BERT-Base', 'BERT-Large']:
                x -= 4  # 向左移2点
            elif model_name in ['GPT-2', 'LLaMA']:
                x -= 2  # 向左移1点
            # T5和ViT保持不变

            ax.text(x, -0.05, model_name,
                    ha='center', va='top', fontsize=28, rotation=45)

    # 添加(bs, seq)标签
    for pair_idx, (bs, seq) in enumerate(BS_SEQ_PAIRS):
        group_start = pair_idx * (group_width + group_spacing)
        x_center = group_start + group_width / 2

        ax.text(x_center, -1.9, f'({bs}, {seq})',
                ha='center', va='top', fontsize=37)


fig, (ax1) = plt.subplots(1, 1, figsize=(20, 3.9), dpi=300, gridspec_kw={'wspace': 0.08})

plot_subplot(ax1, data1_extracted)

legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor='black')
                   for c in COLORS]
fig.legend(legend_elements, LABEL_NAMES,
           loc='upper center',
           ncol=2,
           frameon=False,
           fontsize=37,
           bbox_to_anchor=(0.5, 1.34))

plt.subplots_adjust(wspace=0.12)
plt.savefig('5-Percentage.pdf',
            bbox_inches='tight',
            dpi=300,
            facecolor='white')

plt.close()