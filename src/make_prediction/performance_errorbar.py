import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Your data
data = {
    'Forest Type': [
        'Overall',
        'Woody Savannas', 
        'Savannas', 
        'Deciduous Broadleaf Forest', 
        'Evergreen Broadleaf Forest',
        'Mixed Forest',
        'Evergreen Needleleaf Forest',
        'Shrublands',
    ],
    
    'RF': [0.744, 0.829257, 0.829964, 0.827457, 0.768497, 0.749579, 0.724171, 0.629],
    'XGB': [0.751, 0.835355, 0.83074, 0.785133, 0.76751, 0.754659, 0.727672, 0.638],
    'BPNN': [0.707, 0.798638, 0.787101, 0.712045, 0.698279, 0.666266, 0.638278, 0.545],
    'LSTM': [0.722, 0.814215, 0.797523, 0.784604, 0.744717, 0.708298, 0.693091, 0.532],
    'CNN-LSTM': [0.767, 0.866, 0.848, 0.822, 0.787, 0.766, 0.688, 0.607]
}

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(data)

# Extract model names (excluding 'Forest Type')
models = [col for col in df.columns if col != 'Forest Type']

# Get overall performance (first value in each model column)
overall_performance = [df[model][0] for model in models]

# Calculate min and max for each model (excluding the overall value)
model_mins = [df[model][1:].min() for model in models]
model_maxs = [df[model][1:].max() for model in models]

# Calculate error bar parameters
# Lower error = overall - min, Upper error = max - overall
yerr_lower = [overall - min_val for overall, min_val in zip(overall_performance, model_mins)]
yerr_upper = [max_val - overall for overall, max_val in zip(overall_performance, model_maxs)]
yerr = [yerr_lower, yerr_upper]

# Create the plot
fig, ax = plt.subplots(figsize=(5, 7))

# Set bar positions
x_pos = np.arange(len(models))
width = 0.6

# Create bars with asymmetric error bars
bars = ax.bar(x_pos, overall_performance, width, label='Overall Performance', fill =False,)

# Add error bars
ax.errorbar(x_pos, overall_performance, yerr=yerr, fmt='none', capsize=3, 
            capthick=2, ecolor='black', elinewidth=1)

# Add value labels on top of bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2 + 0.13, height + 0.01,
            f'{overall_performance[i]:.3f}', ha='center', va='bottom', fontsize=10)



# Customize the plot
ax.set_ylabel('r2 Score', fontsize=14)
ax.set_title('r2 Scores for Different Algorithms', fontsize=16)
ax.set_xticks(x_pos)
ax.set_xticklabels(models, fontsize=12)

# save the figure
plt.savefig('performance_errorbar.png', dpi=300, bbox_inches='tight')


plt.tight_layout()
plt.show()