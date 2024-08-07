## Finding the Best Number of Clusters: Elbow Method

**Understanding the Problem:**
We want to figure out how many groups (or clusters) our data should be divided into. A popular method for this is called the "elbow method."

**What We'll Do:**

1. **Try different numbers of groups:** We'll start by trying different numbers of groups for our data.
2. **Measure how well the data fits into each group:** For each number of groups, we'll calculate something called the "Within-Cluster Sum of Squares" (WCSS). This tells us how spread out the data points are within each group. Smaller WCSS is better.
3. **Plot the results:** We'll create a graph showing how the WCSS changes as we increase the number of groups. This graph is called the "elbow curve."
4. **Find the "elbow":** The "elbow" is the point on the graph where the line starts to bend. This is often a good indication of the optimal number of groups.

**Why it works:**

* As we increase the number of groups, the WCSS naturally decreases. This is because each data point will be closer to its assigned group center.
* However, at some point, adding more groups doesn't significantly reduce the WCSS. This is the "elbow" in the curve.
* Beyond the elbow, adding more groups might not provide much benefit, and it could even make the results worse.
By following these steps and understanding the logic behind the elbow method, we can effectively determine the optimal number of clusters for your data.

# Define the range of cluster numbers to evaluate
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

cluster_range = range(1, 11)  # Testing from 1 to 10 clusters
wcss = []
for i in cluster_range:
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)
```

# Plot the Elbow Curve and compute differences to identify the elbow
```python
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, wcss, marker='o', linestyle='-', color='b', label='WCSS')
diffs = np.diff(wcss)
second_diffs = np.diff(diffs)
elbow_index = np.argmin(second_diffs) + 1  # Add 1 to align with cluster numbers
```
# Plot the elbow point
```python
plt.plot(elbow_index, wcss[elbow_index - 1], 'ro')  # Marking the elbow point with red
plt.title('Elbow Method For Optimal Number of Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.xticks(cluster_range)
plt.grid(True)
plt.legend()
plt.show()
print(f'The optimal number of clusters is approximately: {elbow_index}')
```
![output_13_0](https://github.com/user-attachments/assets/44fed43e-e5d4-4dcb-8e51-a5b096b81d25)

```python
# Plot the distribution of clusters
plt.figure(figsize=(10, 6))
sns.countplot(data=df_with_labels, x='Cluster', hue='Cluster', palette='viridis', legend=False)
plt.title('Distribution of Clusters')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.grid(True)
plt.show()
```
![output_20_0](https://github.com/user-attachments/assets/de940235-07db-44d3-8885-85eb2e9da27d)

# Create a pairplot with specified size
```python
height = 5  # Height of each facet
aspect = 2  # Aspect ratio 
pairplot = sns.pairplot(df_with_labels, vars=['tenure', 'MonthlyCharges'], hue='Cluster', palette='viridis', height=height, aspect=aspect)

# Adjust the title position
pairplot.fig.suptitle('Pair Plot of Tenure and Monthly Charges by Cluster', y=1.02)

# Display the plot
plt.show()
```
![output_21_0](https://github.com/user-attachments/assets/91006df6-0d8f-4b41-a8f4-9ae098996e63)

# Numeric columns distrubition by cluster labels
```python
numeric_columns = df_with_labels.select_dtypes(include=['number']).drop("SeniorCitizen", axis=1)
# Group by cluster and calculate the mean of each numeric feature
cluster_centers = df_with_labels.groupby('Cluster')[numeric_columns.columns].mean()
print(cluster_centers)
```

                tenure  MonthlyCharges  Cluster
    Cluster                                    
    0        31.737537       42.028592      0.0
    1        40.984925       63.745854      1.0
    2        16.288148       65.339993      2.0
    3        16.630624       66.249819      3.0
    4        31.507592       87.942896      4.0
    5        56.578875       59.602812      5.0
    

**Distribution of MonthlyCharges by Cluster**
```python
sns.boxplot(x='Cluster', y='MonthlyCharges', data=df_with_labels)
plt.title('Distribution of MonthlyCharges by Cluster')
plt.show()
```
![output_23_0](https://github.com/user-attachments/assets/f09e4a27-0bf2-4441-b392-ca87953c4c18)

**Distribution of Tenure by Cluster**
```python
sns.boxplot(x='Cluster', y='tenure', data=df_with_labels)
plt.title('Distribution of tenure by Cluster')
plt.show()
```
![output_23_1](https://github.com/user-attachments/assets/8ed795a0-b3f1-4798-a398-340b2961fbaa)

# ANN Model visualization on feature of apprpopriate clusters
```python

```

# OTHER MODELS AND WHATIF
```python
# Assuming df_with_labels is your DataFrame with 'Cluster' and 'Churn'
df_cluster_0 = df_with_labels[df_with_labels['Cluster'] == 0]
df_cluster_1 = df_with_labels[df_with_labels['Cluster'] == 1]

# Drop the 'Cluster' column and the target variable 'Churn'
X_0 = df_cluster_0.drop(columns=['Cluster', 'Churn'])
y_0 = df_cluster_0['Churn']

X_1 = df_cluster_1.drop(columns=['Cluster', 'Churn'])
y_1 = df_cluster_1['Churn']

# Create a combined set of unique values from both subsets
all_categorical_columns = set(X_0.select_dtypes(include=['object']).columns).union(X_1.select_dtypes(include=['object']).columns)
label_encoders = {}

# Fit LabelEncoders for each categorical column based on combined unique values
for column in all_categorical_columns:
    le = LabelEncoder()
    all_values = pd.concat([X_0[column], X_1[column]]).unique()
    le.fit(all_values)
    X_0[column] = le.transform(X_0[column])
    X_1[column] = le.transform(X_1[column])
    label_encoders[column] = le

# Train Random Forest models
rf_0 = RandomForestClassifier(n_estimators=100)
rf_0.fit(X_0, y_0)

rf_1 = RandomForestClassifier(n_estimators=100)
rf_1.fit(X_1, y_1)

# Get feature importances
importances_0 = rf_0.feature_importances_
importances_1 = rf_1.feature_importances_

# Create DataFrames for feature importances
features = X_0.columns
importances_df_0 = pd.DataFrame({'Feature': features, 'Importance_Cluster_0': importances_0})
importances_df_1 = pd.DataFrame({'Feature': features, 'Importance_Cluster_1': importances_1})

# Merge dataframes for comparison
importances_df = pd.merge(importances_df_0, importances_df_1, on='Feature')
importances_df = importances_df.sort_values(by='Importance_Cluster_0', ascending=False)
```

**Plot feature importances for Cluster 0**
```python
plt.figure(figsize=(14, 7))
sns.barplot(data=importances_df, x='Importance_Cluster_0', y='Feature', color='blue', orient='h', alpha=0.7)
plt.title('Feature Importances for Cluster 0')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
for index, value in enumerate(importances_df['Importance_Cluster_0']):
    plt.text(value, index, f'{value:.2f}')
plt.show()
```
![output_24_0](https://github.com/user-attachments/assets/9106d615-36d6-4557-aaf1-1de26f2309a9)


**Plot feature importances for Cluster 1**
```python
plt.figure(figsize=(14, 7))
sns.barplot(data=importances_df, x='Importance_Cluster_1', y='Feature', color='green', orient='h', alpha=0.7)
plt.title('Feature Importances for Cluster 1')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
for index, value in enumerate(importances_df['Importance_Cluster_1']):
    plt.text(value, index, f'{value:.2f}')
plt.show()
```
![output_24_1](https://github.com/user-attachments/assets/1b744142-eeef-459c-bfcf-b98ce06ea7da)

```python

```

