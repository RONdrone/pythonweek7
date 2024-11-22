# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv(url, header=None, names=column_names)

# Data Exploration
print("Data Types of Columns:")
print(df.dtypes)

# Observation:
# 'sepal_length', 'sepal_width', 'petal_length', 'petal_width' are numeric (float64),
# while 'species' is a categorical variable (object).
# The data types appear as expected.

print("\nSummary Statistics:")
print(df.describe())

# Observation:
# The summary statistics give us key insights into the distribution of numerical features.
# - Sepal Length: Mean ~ 5.8 cm, range from ~4.3 to ~7.9 cm.
# - Sepal Width: Mean ~ 3.05 cm, range from ~2.0 to ~4.4 cm.
# - Petal Length: Mean ~ 3.8 cm, range from ~1.0 to ~6.9 cm.
# - Petal Width: Mean ~ 1.2 cm, range from ~0.1 to ~2.5 cm.
# The data shows some variability in the measurements across the species.

print("\nChecking for Missing Values:")
print(df.isnull().sum())

# Observation:
# There are no missing values in the dataset, which is good for our analysis.

# Plotting a Histogram for Sepal Length
plt.figure(figsize=(8, 6))
sns.histplot(df['sepal_length'], bins=10, kde=True, color='skyblue')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# Observation:
# The histogram of Sepal Length shows a normal distribution with most data points
# falling between 4.5 cm and 7.5 cm, with a slight peak around 6.0 cm.
# There appears to be no significant skew in the data.

# Pairplot for visualizing relationships between features
sns.pairplot(df, hue='species', diag_kind='hist', markers=["o", "s", "D"])
plt.suptitle('Pairplot of Iris Features', size=16)
plt.show()

# Observation:
# The pairplot provides a great visual representation of the relationships between the features.
# - Sepal Length vs. Petal Length, and Sepal Width vs. Petal Width show clear species separation.
# - Setosa species is very distinct, while Versicolor and Virginica have overlapping feature ranges.
# The pairplot helps in identifying patterns and correlations between features, especially for classification tasks.

# Boxplot of Sepal Length by Species
plt.figure(figsize=(8, 6))
sns.boxplot(x='species', y='sepal_length', data=df, palette='Set2')
plt.title('Boxplot of Sepal Length by Species')
plt.xlabel('Species')
plt.ylabel('Sepal Length (cm)')
plt.show()

# Observation:
# The boxplot shows that Setosa has the smallest Sepal Length compared to the other two species.
# - The median Sepal Length for Setosa is ~5.0 cm.
# - Versicolor and Virginica have larger Sepal Length values, with Versicolor having a median around 5.9 cm
#   and Virginica around 6.5 cm.
# The boxplots highlight the differences in central tendency and variability of Sepal Length between the species.

# Correlation Heatmap of Features
plt.figure(figsize=(8, 6))
sns.heatmap(df.drop('species', axis=1).corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Features')
plt.show()

# Observation:
# The correlation heatmap reveals some interesting patterns:
# - There is a strong positive correlation between 'petal_length' and 'petal_width' (correlation ~ 0.96).
# - There is a moderate positive correlation between 'sepal_length' and 'petal_length' (correlation ~ 0.87).
# - Sepal Width and Petal Width have the weakest correlation (~ 0.37), suggesting that they are less related.
# These correlations may help in feature selection for machine learning models.

# End of Analysis
