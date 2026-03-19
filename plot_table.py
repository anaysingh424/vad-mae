import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(11, 4))
ax.axis('tight')
ax.axis('off')

# Detailed Data Matrix
col_labels = ['Parameter / Metric', 'Our Implementation (RTX 4050)', 'Official Paper Benchmark']
table_data = [
    ['Dataset', 'Avenue Benchmark', 'Avenue Benchmark'],
    ['Hardware Environment', 'NVIDIA GeForce RTX 4050 (6GB)', 'NVIDIA RTX 3090 / A100'],
    ['Framework', 'PyTorch 2.1.0 (CUDA 12.1)', 'PyTorch'],
    ['Training Epochs', '10 (Rapid Evaluation)', '60+ (Full Convergence)'],
    ['Batch Size', '32 (Workers: 8)', '32'],
    ['Peak Macro-AUC', '85.25% (Epoch 4)', '87.20%'],
    ['Peak Micro-AUC', '84.47% (Epoch 3)', 'N/A'],
    ['Training Time', '~31 Minutes', 'Extensive (Projected)']
]

# Generate native Matplotlib table
table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center', edges='horizontal')

# Aesthetic Styling
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.0)

# Colors and Bolding
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white', size=13)
        cell.set_facecolor('#2c3e50')
    elif col == 0:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#ecf0f1')
    else:
        if row % 2 == 0:
            cell.set_facecolor('#f4f6f7')

plt.title('AED-MAE Performance & Environment Matrix', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()

out_path = r'C:\Users\Anay\.gemini\antigravity\brain\de38bd88-82d3-4bc9-bb9b-de31f4d33ba5\auc_comparison_table.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Detailed image table securely saved to {out_path}")
