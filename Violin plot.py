# Re-importing necessary libraries and reloading the data after the reset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the uploaded file again to ensure continuity
data = pd.read_csv('E:/Medicine LLM/figure_coding/缺失值处理_O.csv')

# Preview the data to ensure it's loaded correctly
data.head()

# Extract the necessary columns for the violin plot (from the 4th to the 14th column)
#plot_data = data.iloc[:, 3:24]
plot_data = data.iloc[:, 25:45]

# Define Morandi colors
# Define Nature journal colors
ocean_breeze = sns.color_palette("PuRd", 25).as_hex()
#ocean_breeze = dusk = sns.color_palette("BuPu", 25).as_hex()
#ocean_breeze = citrus = sns.color_palette("YlOrRd", 25).as_hex()


# Initialize a figure with subplots
fig, axes = plt.subplots(nrows=len(plot_data.columns), ncols=1, figsize=(10, 5 * len(plot_data.columns)), constrained_layout=True)

for ax, (column, color) in zip(axes, zip(plot_data.columns, ocean_breeze)):
    # Create a violin plot for each feature
    sns.violinplot(x=plot_data[column], ax=ax, color=color)
    ax.set_title(f'Violin plot of {column}')
    ax.set_xlabel('Values')
    ax.set_ylabel('Density')

plt.show()
