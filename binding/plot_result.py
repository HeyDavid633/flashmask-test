import re
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

def parse_performance_data(file_path):
    data = {}
    current_batch = None
    
    with open(file_path, 'r') as f:
        for line in f:
            # Match batch size line
            batch_match = re.match(r'=== Testing batch_size: (\d+) ===', line)
            if batch_match:
                current_batch = int(batch_match.group(1))
                data[current_batch] = {}
                continue
                
            # Match sequence length line
            seq_match = re.match(r'bs:(\d+) \| h_num:\d+ \| seq:(\d+)', line)
            if seq_match:
                current_seq = int(seq_match.group(2))
                data[current_batch][current_seq] = {}
                continue
                
            # Match performance data
            perf_match = re.match(
                r' bs:\d+ \| seq:\d+ \|  (.+?) : ([\d.]+) ms / iter(?: \|  Speedup/FA2: ([\d.]+))?', 
                line
            )
            if perf_match and current_batch and current_seq:
                method = perf_match.group(1).strip()
                time = float(perf_match.group(2))
                data[current_batch][current_seq][method] = time
                
    return data

def calculate_speedups(data):
    speedups = {}
    for batch_size, seq_data in data.items():
        speedups[batch_size] = {}
        for seq_len, methods in seq_data.items():
            torch_time = methods['Torch Naive']
            speedups[batch_size][seq_len] = {
                method: torch_time / time 
                for method, time in methods.items()
            }
    return speedups

def plot_results(speedups, output_path):
    batch_sizes = sorted(speedups.keys())
    seq_lengths = sorted(next(iter(speedups.values())).keys())
    
    # Set up the figure and subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Attention Mechanism Performance Comparison', fontsize=16)
    
    # Define colors and methods
    colors = {
        'Torch Naive': (1.0, 0.6, 0.2, 0.7),     # Orange
        'FlashAttn2': (0.6, 0.2, 1.0, 0.7),       # Purple
        'FlexAttn': (0.2, 0.6, 1.0, 0.7),         # Blue
        'Bind Kernel': (1.0, 0.2, 0.2, 0.7)       # Red
    }
    methods = ['Torch Naive', 'FlashAttn2', 'FlexAttn', 'Bind Kernel']
    
    # Plot each batch size
    for i, batch_size in enumerate(batch_sizes):
        ax = axes[i]
        x = np.arange(len(seq_lengths))
        width = 0.18  # Width of each bar
        offsets = np.linspace(-1.5*width, 1.5*width, len(methods))
        
        for j, method in enumerate(methods):
            values = [speedups[batch_size][seq][method] for seq in seq_lengths]
            ax.bar(x + offsets[j], values, width, label=method, color=colors[method])
        
        # Customize the plot
        ax.set_title(f'Batch Size = {batch_size}', fontsize=12)
        ax.set_xlabel('Sequence Length', fontsize=10)
        ax.set_ylabel('Speedup (vs Torch Naive)', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(seq_lengths)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        
        # Add horizontal line at y=1 for Torch Naive
        ax.axhline(y=1, color='gray', linestyle='--', linewidth=0.8)
    
    # Add legend to the right of the plots
    fig.legend(
        methods, 
        loc='upper right',
        bbox_to_anchor=(0.98, 0.9),
        ncol=1,
        fontsize=10
    )
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Plot attention mechanism performance results.')
    parser.add_argument('input_file', help='Path to the performance results text file')
    args = parser.parse_args()
    
    # Determine output path
    input_basename = os.path.basename(args.input_file)
    output_filename = os.path.splitext(input_basename)[0] + '.pdf'
    output_dir = os.path.join(os.getcwd(), 'plot_result')
    output_path = os.path.join(output_dir, output_filename)
    
    # Process data and plot
    data = parse_performance_data(args.input_file)
    speedups = calculate_speedups(data)
    plot_results(speedups, output_path)

if __name__ == '__main__':
    main()