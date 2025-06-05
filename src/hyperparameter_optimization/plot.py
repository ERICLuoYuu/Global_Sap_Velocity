import matplotlib.pyplot as plt
import numpy as np

# Data from the scatter plots
data = {
    'Woody Savannas': {'r2': 0.866, 'n': 4035, 'is_forest': True},
    'Savannas': {'r2': 0.848, 'n': 5649, 'is_forest': False},
    'Deciduous Broadleaf Forest': {'r2': 0.822, 'n': 6994, 'is_forest': True},
    'Evergreen Broadleaf Forest': {'r2': 0.787, 'n': 8608, 'is_forest': True},
    'Mixed Forest': {'r2': 0.766, 'n': 4035, 'is_forest': True},
    'Evergreen Needleleaf Forest': {'r2': 0.688, 'n': 6456, 'is_forest': True},
    'Shrublands': {'r2': 0.607, 'n': 1883, 'is_forest': False}
}

# Sort by R² value
sorted_types = sorted(data.items(), key=lambda x: x[1]['r2'])
types_labels = [item[0] for item in sorted_types]
r2_values = [item[1]['r2'] for item in sorted_types]
n_values = [item[1]['n'] for item in sorted_types]
is_forest = [item[1]['is_forest'] for item in sorted_types]

# Create the figure and axis
fig, ax = plt.subplots(figsize=(6, 8))  # Adjusted figure size for vertical orientation

# Create vertical bar chart
bars = ax.bar(range(len(types_labels)), r2_values, color='forestgreen', edgecolor='black', alpha=0.4, width=0.9)

# Customize the plot
ax.set_ylabel('R² Score', fontsize=14, fontweight='bold')
ax.set_xlabel('Forest Type', fontsize=14, fontweight='bold')
ax.set_title('CNN-LSTM Model Performance by Forest Type', fontsize=10, fontweight='bold', pad=20)

# Set x-axis labels
ax.set_xticks(range(len(types_labels)))
ax.set_xticklabels(types_labels, fontsize=12, rotation=45, ha='right')

# Set y-axis limits
ax.set_ylim(0, 1.0)
ax.set_yticks(np.arange(0, 1.1, 0.1))

# Add value labels on the bars
for i, (bar, r2, n) in enumerate(zip(bars, r2_values, n_values)):
    # R² value
    ax.text(bar.get_x() + bar.get_width()/2, r2 + 0.01, 
            f'{r2:.3f}', 
            ha='center', va='bottom', fontsize=12, fontweight='bold')
    


# Add grid
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Set background color for better contrast
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('igbp_r2_vertical_barchart.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()