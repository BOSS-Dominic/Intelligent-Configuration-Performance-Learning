import re
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define file paths relative to the script directory
# Ensure 'Decision tree results.txt' and 'Random Forest results.txt' are in the same folder as this script
dt_file_path = os.path.join(script_dir, 'Decision tree results.txt')
rf_file_path = os.path.join(script_dir, 'Random Forest results.txt')

# 1. Initialize data structure to store MAPE lists for each system
data = defaultdict(lambda: {'LR': [], 'DT': [], 'RF': []})

# 2. Parse Decision Tree (DT) results file
try:
    with open(dt_file_path, 'r', encoding='utf-8') as f:
        current_system = None
        for line in f:
            if '> System:' in line:
                # Extract system name
                m = re.search(r'> System:\s*([^,]+),', line)
                if m:
                    current_system = m.group(1).strip()
            elif '【Average MAPE】' in line and current_system:
                # Extract LR and DT MAPE values
                m = re.search(r'Baseline\(LR\):\s*([\d\.]+)\s*\|\s*New tool\(DT\):\s*([\d\.]+)', line)
                if m:
                    lr_val = float(m.group(1))
                    dt_val = float(m.group(2))
                    data[current_system]['LR'].append(lr_val)
                    data[current_system]['DT'].append(dt_val)
except FileNotFoundError:
    print(f"Error: File not found - {dt_file_path}")

# 3. Parse Random Forest (RF) results file
try:
    with open(rf_file_path, 'r', encoding='utf-8') as f:
        current_system = None
        for line in f:
            if '> System:' in line:
                m = re.search(r'> System:\s*([^,]+),', line)
                if m:
                    current_system = m.group(1).strip()
            elif '【Average MAPE】' in line and current_system:
                # Extract RF MAPE value
                m = re.search(r'Baseline\(LR\):\s*([\d\.]+)\s*\|\s*New tool\(RF\):\s*([\d\.]+)', line)
                if m:
                    rf_val = float(m.group(2))
                    data[current_system]['RF'].append(rf_val)
except FileNotFoundError:
    print(f"Error: File not found - {rf_file_path}")

# 4. Calculate average MAPE for all datasets under each system
systems = list(data.keys())

if systems:
    lr_avgs = [np.mean(data[s]['LR']) for s in systems]
    dt_avgs = [np.mean(data[s]['DT']) for s in systems]
    rf_avgs = [np.mean(data[s]['RF']) for s in systems]

    # 5. Start plotting the comparison bar chart
    x = np.arange(len(systems))
    width = 0.25 

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot three groups of bars
    rects1 = ax.bar(x - width, lr_avgs, width, label='Baseline (LR)', color='#e74c3c', edgecolor='black')
    rects2 = ax.bar(x, dt_avgs, width, label='Decision Tree (DT)', color='#f1c40f', edgecolor='black')
    rects3 = ax.bar(x + width, rf_avgs, width, label='Random Forest (RF)', color='#2980b9', edgecolor='black')

    # Chart decorations
    ax.set_ylabel('Average MAPE', fontsize=12, fontweight='bold')
    ax.set_title('Overall Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in systems], fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)

    # Automatically label values
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',  
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), # Vertical offset upwards
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()

    # Save the output image in the same directory as the script
    output_path = os.path.join(script_dir, 'system_comparison_3bars.png')
    plt.savefig(output_path, dpi=300)
    plt.show()
else:
    print("No data was parsed. Please check the input files and their format.")