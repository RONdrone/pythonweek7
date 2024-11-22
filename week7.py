# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1: Load and Explore the Dataset

# Step 1: Load the dataset
# For this example, we use the Iris dataset (you can replace it with your dataset path).
try:
    df = pd.read_csv("iris.csv")  # Replace with your dataset file path
except FileNotFoundError:
    print("Error: The file was not found. Please check the file path.")
    exit()

# Step 2: Display the first few rows of the dataset to inspect it
print("First 5 rows of the dataset:")
print(df.head())

# Observation:
# The first 5 rows show that the dataset contains 5 columns:
# - sepal_length, sepal_width, petal_length, petal_width (all numerical features)
# - species (categorical target variable)
# Each row corresponds to an observation of one flower, with measurements of its sepals and petals, and its species.

# Step 3: Explore the structure of the dataset
# Check data types and missing values
print("\nData types of columns:")
print(df.dtypes)

# Observation:
# The data types indicate that the numerical columns (sepal_length, sepal_width, petal_length, petal_width) are of type float64.
# The species column is of type object, which is expected since it's categorical.

print("\nChecking for missing values:")
print(df.isnull().sum())

# Observation:
# There are no missing values in the dataset, meaning we do not need to handle missing data. This simplifies the analysis.

# Step 4: Clean the dataset
# Drop rows with missing values (if any). In this case, we don't need this step as there are no missing values.
df = df.dropna()

# Check if missing values are gone
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Task 2: Basic Data Analysis

# Step 5: Compute basic statistics (mean, median, standard deviation) for numerical columns
print("\nSummary statistics of numerical columns:")
print(df.describe())

# Observation:
# The summary statistics give us a sense of the central tendency and spread of each numerical feature.
# - Sepal Length: Mean = 5.84 cm, Min = 4.3 cm, Max = 7.9 cm.
# - Sepal Width: Mean = 3.05 cm, Min = 2.0 cm, Max = 4.4 cm.
# - Petal Length: Mean = 3.76 cm, Min = 1.0 cm, Max = 6.9 cm.
# - Petal Width: Mean = 1.2 cm, Min = 0.1 cm, Max = 2.5 cm.
# These statistics help us understand the range and distribution of the features.

# Step 6: Perform groupings on a categorical column
# We'll group by 'species' (since Iris dataset has a species column) and compute the mean of numerical columns for each group
grouped = df.groupby('species').mean()
print("\nAverage measurements per species:")
print(grouped)

# Observation:
# The mean values for each species show the differences in sepal and petal measurements:
# - Setosa: Has the smallest measurements for both sepals and petals.
# - Versicolor: Intermediate values.
# - Virginica: Has the largest measurements, particularly in petal length and width.

# Task 3: Data Visualization

# Step 7: Create at least four different types of visualizations

# Line chart (If the dataset contains time-series data, you could plot trends over time)
# For this example, the Iris dataset does not contain time-series data, so we just show a placeholder line chart.
plt.figure(figsize=(10, 6))
sns.lineplot(x=df.index, y=df['sepal_length'], label='Sepal Length')
plt.title('Line Chart: Sepal Length Trend')
plt.xlabel('Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.show()

# Observation:
# The line chart shows how `sepal_length` changes across the dataset's rows (observations). Since the dataset is not time-series, 
# the line chart isn't very informative here, but it could be useful if we had data over time.

# Bar chart (Average sepal length for each species)
plt.figure(figsize=(8, 6))
sns.barplot(x='species', y='sepal_length', data=df, palette='Set2')
plt.title('Bar Chart: Average Sepal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Sepal Length (cm)')
plt.show()

# Observation:
# The bar chart shows that Setosa has the smallest average sepal length (~5.0 cm), while Virginica has the largest (~6.5 cm).
# Versicolor falls in between. This indicates that the sepal length is a good distinguishing feature for the species.

# Histogram (Distribution of sepal length)
plt.figure(figsize=(8, 6))
sns.histplot(df['sepal_length'], bins=10, kde=True, color='skyblue')
plt.title('Histogram: Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# Observation:
# The histogram shows that `sepal_length` has a normal distribution with a slight peak around 6 cm.
# Most of the data points lie between 4.5 and 7.5 cm, indicating that Sepal Length is fairly uniform across species.

# Scatter plot (Visualize relationship between sepal length and petal length)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal_length', y='petal_length', data=df, hue='species', style='species', palette='Set1')
plt.title('Scatter Plot: Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()

# Observation:
# The scatter plot clearly shows that Sepal Length and Petal Length are correlated, especially for Versicolor and Virginica.
# Setosa is distinct, with both shorter sepal and petal lengths. The plot highlights the potential for using sepal and petal dimensions
# for classification tasks.

