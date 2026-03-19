import json
import matplotlib.pyplot as plt  # type: ignore
import os

log_test_path = 'experiments/avenue/log_test.txt'

epochs = []
micro_aucs = []
macro_aucs = []

with open(log_test_path, 'r') as f:
    lines = f.readlines()

# The real training loop consists of the final 10 logged lines
real_data = [lines[i] for i in range(max(0, len(lines) - 10), len(lines))]

for idx, line in enumerate(real_data):
    if not line.strip():
        continue
    data = json.loads(line.strip())
    epochs.append(idx + 1)  # 1 to 10
    micro_aucs.append(data['test_micro'] * 100)
    macro_aucs.append(data['test_macro'] * 100)

# Styling and plot generation
plt.style.use('ggplot')
plt.figure(figsize=(10, 6))

plt.plot(epochs, macro_aucs, marker='o', markersize=8, linewidth=2.5, label='Macro-AUC (Our RTX 4050 Run)', color='#2ecc71')
plt.plot(epochs, micro_aucs, marker='s', markersize=8, linewidth=2.5, label='Micro-AUC (Our RTX 4050 Run)', color='#3498db')

# Add paper benchmark line
plt.axhline(y=87.2, color='#e74c3c', linestyle='--', linewidth=2.5, label='Target Paper Benchmark (87.2%)')

plt.title('AED-MAE Validation AUC over 10 Epochs (Avenue Dataset)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Training Epochs', fontsize=12, fontweight='bold')
plt.ylabel('AUC Score (%)', fontsize=12, fontweight='bold')

plt.xticks(epochs)
plt.ylim(75, 90) # Framing the scores tightly around 76-87
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=11, loc='lower right', frameon=True, shadow=True)

plt.tight_layout()
out_path = r'C:\Users\Anay\.gemini\antigravity\brain\de38bd88-82d3-4bc9-bb9b-de31f4d33ba5\auc_training_graph.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Graph successfully saved to {out_path}")
