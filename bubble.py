import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the specified path
data = pd.read_csv('E:\\papers\\Xu_project1\\code\\Medicine LLM\\figure_coding\\keyfactors_RA.csv')

# Set global font to Times New Roman for ticks
plt.rcParams['font.family'] = 'Times New Roman'

# Plotting the bubble chart with a rainbow color map
plt.figure(figsize=(16, 10))  # Increased figure size for better readability
scatter = plt.scatter(
    data['Feature'], data['Importance'],
    s=data['label']*10, c=range(len(data['Feature'])),
    cmap='rainbow', alpha=0.5
)

# Add colorbar
plt.colorbar(scatter)

# Set labels and title with Times New Roman font
plt.xlabel('Feature', fontsize=25, fontname='Times New Roman')
plt.ylabel('Importance', fontsize=25, fontname='Times New Roman')
plt.title('Bubble Chart of Features by Importance of RA', fontsize=30, fontname='Times New Roman')

# Rotate x-axis labels to 45 degrees and adjust font size to avoid overlap
plt.xticks(rotation=45, fontsize=30, fontname='Times New Roman', ha='right')
plt.yticks(fontsize=14, fontname='Times New Roman')

# Add grid and adjust layout
plt.grid(True)
plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels

# Save the plot to a file
plt.savefig('E:\\papers\\Xu_project1\\code\\Medicine LLM\\figure_coding\\bubble_chart_RA1.png', dpi=300)

# Show the plot
plt.show()
