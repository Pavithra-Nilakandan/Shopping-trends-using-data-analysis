1. Data Cleaning Visualization

import pandas as pd
import matplotlib.pyplot as plt

# Sample dataset before cleaning
data_raw = {'Customer_ID': [101, 102, 103, None, 105],
            'Purchase_Amount': [200, 300, None, 400, 500],
            'Category': ['Electronics', None, 'Clothing', 'Clothing', 'Electronics']}
df_raw = pd.DataFrame(data_raw)

# Cleaning Data
df_cleaned = df_raw.dropna().fillna({'Category': 'Unknown'})

# Visualizing Missing Values
plt.figure(figsize=(8, 4))
plt.bar(['Raw Data', 'Cleaned Data'], [df_raw.isnull().sum().sum(), df_cleaned.isnull().sum().sum()], color=['red', 'green'])
plt.title('Data Cleaning: Missing Values Resolved')
plt.ylabel('Count of Missing Values')
plt.show()

2. EDA - Trends and Preferences

import seaborn as sns

# Sample dataset for visualization
data = {'Category': ['Electronics', 'Clothing', 'Groceries', 'Electronics', 'Groceries'],
        'Sales': [200, 150, 300, 400, 500],
        'Month': ['Jan', 'Jan', 'Feb', 'Feb', 'Mar']}
df = pd.DataFrame(data)

# Bar Plot for Category Sales
plt.figure(figsize=(8, 5))
sns.barplot(x='Category', y='Sales', data=df, palette='Blues_d')
plt.title('Sales by Category')
plt.show()

# Line Plot for Monthly Trends
plt.figure(figsize=(8, 5))
sns.lineplot(x='Month', y='Sales', hue='Category', data=df, marker='o')
plt.title('Monthly Sales Trends')
plt.show()
3. Machine Learning Results - Clustering

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np

# Generating synthetic data
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# Applying K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Plotting Clustering Results
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
plt.title('Customer Segmentation Using K-Means Clustering')
plt.show()

