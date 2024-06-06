import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
# Replace 'your_file_path.csv' with the path to your CSV file
df = pd.read_csv('/Users/astrid/PycharmProjects/thesis-work/plots_analysis/crepe_pitch_may-24.csv')

def assign_age_group(age, age_groups):
    for group_name, age_range in age_groups.items():
        if age_range[0] <= age < age_range[1]:
            return group_name
    return 'Unknown'

# Define age groups
age_groups = {
    'kitten': (0, 0.5),
    'adult': (0.5, 12),
    'senior': (12, 20)
}

# Create a new column for the age group
df['age_group'] = df['Age'].apply(assign_age_group, age_groups=age_groups)

# Assigning a unique color to each age group
age_group_colors = {
    "kitten": "blue",
    "adult": "red",
    "senior": "purple"
}

# Adding a color column based on the 'age_group' column
df['Color'] = df['age_group'].map(age_group_colors)

# Creating the scatter plot
plt.figure(figsize=(10, 6))

# Plotting each age group separately to have different colors
for age_group, color in age_group_colors.items():
    subset = df[df['age_group'] == age_group]
    plt.scatter(subset.index, subset['Mean F0'], s=100, c=color, label=age_group)

plt.title('Mean Frequency by Age Group')
plt.xlabel('Sample')
plt.ylabel('Mean Frequency (Hz)')
plt.legend(title="Age Group")
plt.grid(True)

plt.show()
