import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Create DataFrame from the data
data = {
    'Forest Type': [
        'overall'
        'Woody Savannas', 
        'Savannas', 
        'Deciduous Broadleaf Forest', 
        'Evergreen Broadleaf Forest',
        
        'Mixed Forest',
        'Evergreen Needleleaf Forest',
        'Shrublands',
    ],
    'ANN': [0.707, 0.798638, 0.787101, 0.712045, 0.698279,  0.666266, 0.638278, 0.545,],
    'RF': [0.744, 0.839257, 0.829964, 0.827457, 0.768497, 0.749579, 0.724171, 0.689],
    'XGB': [0.751, 0.835355, 0.83074, 0.785133, 0.76751,  0.754659, 0.727672, 0.668,],
    'LSTM': [0.722, 0.814215, 0.797523, 0.784604, 0.744717,  0.708298, 0.693091, 0.532,],
    'CNN-LSTM': [0.767, 0.866, 0.848, 0.822, 0.787,  0.766, 0.688, 0.607,]
}

df = pd.DataFrame(data)
df = df.set_index('Forest Type')

# Transpose the DataFrame to have models on x-axis
df_transposed = df.T

# Define the forest colors
forest_colors = {
    "ENF - Evergreen Needleleaf": "#2E8B57",  # Sea Green
    "EBF - Evergreen Broadleaf": "#228B22",   # Forest Green
    "DNF - Deciduous Needleleaf": "#6B8E23",  # Olive Drab
    "DBF - Deciduous Broadleaf": "#32CD32",   # Lime Green
    "MF - Mixed Forest": "#3CB371",           # Medium Sea Green
    "WSA - Woody Savannas": "#8FBC8F",        # Dark Sea Green
    "SAV - Savannas": "#9ACD32", 
    # add one for Shrublands
    "SHR - Shrublands":  "#808000"  
}

# Create a mapping from the dataframe column names to the color dict keys
color_mapping = {
    'Woody Savannas': 'WSA - Woody Savannas',
    'Savannas': 'SAV - Savannas',
    'Deciduous Broadleaf Forest': 'DBF - Deciduous Broadleaf',
    'Evergreen Broadleaf Forest': 'EBF - Evergreen Broadleaf',
    'Mixed Forest': 'MF - Mixed Forest',
    'Evergreen Needleleaf Forest': 'ENF - Evergreen Needleleaf',
    'Shrublands': 'SHR - Shrublands'  
}

# Create the figure and axes
fig, ax = plt.subplots(figsize=(14, 10))

# Get the width of each bar
width = 0.75

# Set up positions for the bars
x = np.arange(len(df_transposed.index))

# Plot the bars for each forest type as a grouped bar chart
bar_positions = {}
for i, forest in enumerate(df_transposed.columns):
    bars = ax.bar(x, df_transposed[forest], width, 
                  label=color_mapping[forest], 
                  color=forest_colors[color_mapping[forest]])

    # Store the positions of the bars for later labeling
    bar_positions[forest] = bars

# Set the x-axis labels to be the model names
ax.set_xticks(x)
# set y limits to 0.6-1

ax.set_ylim(0.4, 1.0)
ax.set_xticklabels(df_transposed.index)

# Add labels and title
ax.set_xlabel('Model Type', fontsize=14)
ax.set_ylabel('R² Value', fontsize=14)
ax.set_title('R² Values of Different Forest Types by Model', fontsize=16)

# Add grid lines
ax.grid(True, axis='y', linestyle='--', alpha=0.3)

# Add a legend
ax.legend(title='Forest Types', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('r2_by_model_forest_type.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()