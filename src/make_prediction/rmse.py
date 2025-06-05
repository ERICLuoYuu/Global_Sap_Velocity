import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Original performance data
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
    'ANN': [0.707, 0.798638, 0.787101, 0.712045, 0.698279, 0.666266, 0.638278, 0.545],
    'RF': [0.744, 0.839257, 0.829964, 0.827457, 0.768497, 0.749579, 0.724171, 0.629],
    'XGB': [0.751, 0.835355, 0.83074, 0.785133, 0.76751, 0.754659, 0.727672, 0.638],
     'BPNN': [0.707, 0.798638, 0.787101, 0.712045, 0.698279, 0.666266, 0.638278, 0.545],
    'LSTM': [0.722, 0.814215, 0.797523, 0.784604, 0.744717, 0.708298, 0.693091, 0.532],
    'CNN-LSTM': [0.767, 0.866, 0.848, 0.822, 0.787, 0.766, 0.688, 0.607]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Given overall RMSE for ANN
ann_overall_rmse = 3.075

# Calculate overall RMSE for each model based on performance ratios
models = [ 'RF', 'XGB', 'BPNN', 'LSTM', 'CNN-LSTM']
overall_rmse = {
    
    'RF': ann_overall_rmse * (df['ANN'][0] / df['RF'][0]),
    'XGB': ann_overall_rmse * (df['ANN'][0] / df['XGB'][0]),
    'BPNN': ann_overall_rmse,
    'LSTM': ann_overall_rmse * (df['ANN'][0] / df['LSTM'][0]),
    'CNN-LSTM': ann_overall_rmse * (df['ANN'][0] / df['CNN-LSTM'][0])
}

# Calculate RMSE for each model and forest type
rmse_data = {'Forest Type': df['Forest Type'].tolist()}

for model in models:
    rmse_values = []
    for i, forest_type in enumerate(df['Forest Type']):
        if i == 0:  # Overall
            rmse_values.append(overall_rmse[model])
        else:
            # Calculate RMSE for this forest type based on performance ratio
            performance_ratio = df[model][0] / df[model][i]
            rmse_values.append(overall_rmse[model] * performance_ratio)
    rmse_data[model] = rmse_values

# Create DataFrame with RMSE values
rmse_df = pd.DataFrame(rmse_data)

# Extract min and max RMSE for each model (excluding overall value)
min_rmse = {model: min(rmse_df[model][1:]) for model in models}
max_rmse = {model: max(rmse_df[model][1:]) for model in models}

# Calculate error bar parameters
# Lower error = overall - min, Upper error = max - overall
overall_values = [overall_rmse[model] for model in models]
lower_errors = [overall_rmse[model] - min_rmse[model] for model in models]
upper_errors = [max_rmse[model] - overall_rmse[model] for model in models]
asymmetric_error = [lower_errors, upper_errors]

# Create the barplot with error bars
fig, ax = plt.subplots(figsize=(5, 7))

# Set bar positions
x_pos = np.arange(len(models))
width = 0.6

# Create bars with asymmetric error bars
bars = ax.bar(x_pos, overall_values, width, label='Overall RMSE', fill =False)

# Add error bars
ax.errorbar(x_pos, overall_values, yerr=asymmetric_error, fmt='none', capsize=3, 
            capthick=2, ecolor='black', elinewidth=1)

# Add value labels on top of bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
            f'{overall_values[i]:.3f}', ha='center', va='bottom', fontsize=12)



# Customize the plot
ax.set_ylabel('RMSE', fontsize=14)
ax.set_title('RMSE for Different Algorithms', fontsize=16)
ax.set_xticks(x_pos)
ax.set_xticklabels(models, fontsize=12)

# Set y-axis limits with some padding
min_y = min([min_rmse[model] for model in models]) - 0.3
max_y = max([max_rmse[model] for model in models]) + 0.3
# save the figure as a png file
plt.savefig('rmse_by_model.png', dpi=300, bbox_inches='tight')








plt.tight_layout()
plt.show()