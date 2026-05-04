import matplotlib.pyplot as plt
import csv

# Read data
mileages, prices, linear_preds, probs, labels = [], [], [], [], []
with open('output_data.csv') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        mileages.append(float(row[0]))
        prices.append(float(row[1]))
        linear_preds.append(float(row[2]))
        probs.append(float(row[3]))
        labels.append(int(row[4]))

# Sort by mileage so lines are smooth
sorted_data = sorted(zip(mileages, prices, linear_preds, probs, labels), key=lambda x: x[0])
mileages, prices, linear_preds, probs, labels = zip(*sorted_data)

# Create figure with two y-axes
fig, ax1 = plt.subplots(figsize=(12, 6))

# --- Left y-axis: Price ---
ax1.set_xlabel('Mileage (km)')
ax1.set_ylabel('Price ($)', color='black')
ax1.scatter(mileages, prices, alpha=0.5, c='black', s=20, zorder=5)
ax1.plot(mileages, linear_preds, color='red', linewidth=2, label='Linear Regression (price)')
ax1.tick_params(axis='y', labelcolor='black')

# --- Right y-axis: Probability ---
ax2 = ax1.twinx()
ax2.set_ylabel('Probability (expensive)', color='blue')
ax2.plot(mileages, probs, color='blue', linewidth=2, label='Logistic Regression (prob)')
ax2.axhline(y=0.5, color='blue', linestyle='--', alpha=0.4, label='Decision boundary (0.5)')
ax2.tick_params(axis='y', labelcolor='blue')

# Color-code scatter points by actual label
for i in range(len(mileages)):
    color = 'green' if labels[i] == 1 else 'orange'
    ax1.scatter(mileages[i], prices[i], c=color, s=40, zorder=6, alpha=0.8)

# Legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='red', linewidth=2, label='Linear Regression'),
    Line2D([0], [0], color='blue', linewidth=2, label='Logistic Regression (prob)'),
    Line2D([0], [0], color='blue', linestyle='--', alpha=0.4, label='Decision Boundary (0.5)'),
    Patch(facecolor='green', alpha=0.8, label='Actually Expensive'),
    Patch(facecolor='orange', alpha=0.8, label='Actually Cheap'),
]
ax1.legend(handles=legend_elements, loc='upper right')

plt.title('Linear & Logistic Regression on Car Data')
plt.tight_layout()
plt.savefig('visualisation.png', dpi=150)
plt.show()
print("Plot saved as visualisation.png")