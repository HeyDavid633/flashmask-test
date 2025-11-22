import matplotlib.pyplot as plt
import numpy as np

# 数据整理
configs = [
    {'name': 'bs=1 | seq=128', 'models': {
        'bert_small': {'Torch Native': 3.406, 'Torch Compile': 1.874, 'ByteTransformer': 2.776, 'TileLang': 2.669, 'STOF': 3.282},
        'bert_base': {'Torch Native': 9.226, 'Torch Compile': 3.596, 'ByteTransformer': 5.522, 'TileLang': 6.824, 'STOF': 8.235},
        'bert_large': {'Torch Native': 14.342, 'Torch Compile': 8.965, 'ByteTransformer': 11.020, 'TileLang': 10.838, 'STOF': 16.362},
        'gpt': {'Torch Native': 8.670, 'Torch Compile': 3.673, 'ByteTransformer': 9.498, 'TileLang': 7.222, 'STOF': 8.140},
        't5': {'Torch Native': 18.992, 'Torch Compile': 6.210, 'ByteTransformer': 15.967, 'TileLang': 14.887, 'STOF': 18.692}
    }},
    {'name': 'bs=8 | seq=512', 'models': {
        'bert_small': {'Torch Native': 3.722, 'Torch Compile': 2.705, 'ByteTransformer': 3.705, 'TileLang': 2.631, 'STOF': 3.152},
        'bert_base': {'Torch Native': 10.784, 'Torch Compile': 7.087, 'ByteTransformer': 9.227, 'TileLang': 6.014, 'STOF': 6.436},
        'bert_large': {'Torch Native': 33.112, 'Torch Compile': 22.046, 'ByteTransformer': 27.317, 'TileLang': 19.095, 'STOF': 19.580},
        'gpt': {'Torch Native': 12.543, 'Torch Compile': 7.097, 'ByteTransformer': 11.199, 'TileLang': 7.256, 'STOF': 8.294},
        't5': {'Torch Native': 31.086, 'Torch Compile': 13.413, 'ByteTransformer': 26.548, 'TileLang': 16.253, 'STOF': 18.639}
    }},
    {'name': 'bs=16 | seq=2048', 'models': {
        'bert_small': {'Torch Native': 92.142, 'Torch Compile': 39.695, 'TileLang': 16.184, 'STOF': 17.189},
        'bert_base': {'Torch Native': 282.675, 'Torch Compile': 127.479, 'TileLang': 59.344, 'STOF': 62.344},
        'bert_large': {'Torch Native': 773.815, 'TileLang': 180.610, 'STOF': 188.672},
        'gpt': {'Torch Native': 323.816, 'Torch Compile': 127.603, 'TileLang': 88.869, 'STOF': 63.030},
        't5': {'Torch Native': 829.174, 'TileLang': 157.564, 'STOF': 169.158}
    }}
]

# 颜色映射
framework_colors = {
    'Torch Native': '#1f77b4',
    'Torch Compile': '#ff7f0e',
    'ByteTransformer': '#2ca02c',
    'TileLang': '#9467bd',
    'STOF': '#d62728'
}

# 创建图表
fig, axes = plt.subplots(1, 3, figsize=(24, 6))
plt.subplots_adjust(wspace=0.3, hspace=0.4)

for ax_idx, config in enumerate(configs):
    ax = axes[ax_idx]
    models = list(config['models'].keys())
    
    # 获取所有框架并按特定顺序排列（STOF放在最后）
    all_frameworks = set().union(*(d.keys() for d in config['models'].values()))
    framework_order = ['Torch Native', 'Torch Compile', 'ByteTransformer', 'TileLang', 'STOF']
    frameworks = [fw for fw in framework_order if fw in all_frameworks]
    
    # 计算每个模型的框架数量
    n_frameworks = len(frameworks)
    bar_width = 0.15
    index = np.arange(len(models))
    
    # 绘制每个框架的柱状图（STOF最后绘制）
    for i, fw in enumerate(frameworks):
        values = []
        for model in models:
            values.append(config['models'][model].get(fw, np.nan))
        
        ax.bar(index + i * bar_width, values, bar_width, 
               label=fw, color=framework_colors[fw])
    
    # 设置图表属性
    ax.set_title(f'Configuration: {config["name"]}', fontsize=14)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Time per Iteration (ms)', fontsize=12)
    ax.set_xticks(index + bar_width * (n_frameworks-1)/2)
    ax.set_xticklabels(models, fontsize=11)
    
    # 根据配置调整Y轴范围
    if ax_idx == 0:  # bs=1 | seq=128
        ax.set_ylim(0, 25)
    elif ax_idx == 1:  # bs=8 | seq=512
        ax.set_ylim(0, 40)
    else:  # bs=16 | seq=2048
        ax.set_ylim(0, 900)
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(fontsize=10)

# 添加总标题
plt.suptitle('Deep Learning Framework Performance Comparison', fontsize=18, y=1.0)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留出空间
plt.savefig('framework_performance_comparison.png', bbox_inches='tight')
