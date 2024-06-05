import matplotlib.pyplot as plt
import pandas as pd

# Prepare the data
data = {
    'Metric': ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Model G': [0.113, 0, 0.545, 0.081, 0.343],
    'Model J': [0.005, 0.000, 0.303, 0.026, 0.025],
    'Model K': [0.72, 0.857, 0.486, 0.83, 0.856]
}

df = pd.DataFrame(data)

# Plot the data
fig, ax = plt.subplots(figsize=(10, 6))

df.set_index('Metric').plot(kind='bar', ax=ax, color=['salmon', 'lightgreen', 'skyblue'])

# Adding labels and title
plt.xlabel('Metric', fontweight='bold')
plt.ylabel('Levene Test P-value', fontweight='bold')
plt.title('Levene Test P-values for Models G, J, and K Across Metrics', fontweight='bold')
plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Level (0.05)')
plt.legend(title='Model')
plt.grid(axis='y')

# Set x-axis labels to horizontal
ax.set_xticklabels(df['Metric'], rotation=0)

# Show the plot
plt.show()
