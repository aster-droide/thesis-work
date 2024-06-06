import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np


# Load the CSV file
df = pd.read_csv('/Users/astrid/PycharmProjects/thesis-work/plots_analysis/crepe_pitch_may-24.csv')

# Sort the DataFrame by age
df.sort_values(by='Age', inplace=True)

plt.scatter(df['Age'], df['Mean F0'])
plt.title('Change in Fundamental Frequency (F0) Over Age')
plt.xlabel('Age (Years)')
plt.ylabel('Mean F0 (Hz)')
plt.grid(True)
plt.show()

# density estimation
sns.kdeplot(data=df, x='Age', y='Mean F0')
plt.title('Density Estimate of Mean F0 Over Age')
plt.xlabel('Age (Years)')
plt.ylabel('Mean F0 (Hz)')
plt.show()

# Apply a simple moving average with a window of size N
N = 5  # window size can be changed as needed
df['Mean F0 Smoothed'] = df['Mean F0'].rolling(window=N).mean()

# Plotting the smoothed data
plt.scatter(df['Age'], df['Mean F0'], alpha=0.5, label='Original Data')
plt.plot(df['Age'], df['Mean F0 Smoothed'], color='red', label='Smoothed Data')
plt.title('Change in Fundamental Frequency (F0) Over Age (Smoothed)')
plt.xlabel('Age (Years)')
plt.ylabel('Mean F0 (Hz)')
plt.legend()
plt.grid(True)
plt.show()




# Apply LOESS smoothing (fraction determines the degree of smoothing)
loess_smoothed = lowess(df['Mean F0'], df['Age'], frac=0.1)

# Plotting the smoothed data
plt.scatter(df['Age'], df['Mean F0'], alpha=0.5, label='Original Data')
plt.plot(loess_smoothed[:, 0], loess_smoothed[:, 1], color='red', label='LOESS Smoothed Data')
plt.title('Change in Fundamental Frequency (F0) Over Age (LOESS Smoothed)')
plt.xlabel('Age (Years)')
plt.ylabel('Mean F0 (Hz)')
plt.legend()
plt.grid(True)
plt.show()



# Fit a linear trend line
z = np.polyfit(df['Age'], df['Mean F0'], 1)
p = np.poly1d(z)

# Plotting the trend line
plt.scatter(df['Age'], df['Mean F0'], alpha=0.5)
plt.plot(df['Age'], p(df['Age']), "r--", label='Trend Line')
plt.title('Change in Fundamental Frequency (F0) Over Age with Trend Line')
plt.xlabel('Age (Years)')
plt.ylabel('Mean F0 (Hz)')
plt.legend()
plt.grid(True)
plt.show()



# Calculate the standard deviation of the residuals
residuals = df['Mean F0'] - df['Mean F0 Smoothed']
std_dev = np.std(residuals)

# Consider points that are more than 2 standard deviations from the mean as potential outliers
outliers = df[np.abs(residuals) > 2 * std_dev]

# Show potential outliers
print(outliers[['Filename', 'Age', 'Mean F0']])



