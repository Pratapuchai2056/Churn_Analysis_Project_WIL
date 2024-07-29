**1. Introduction**
The popular clustering process called K-Means clustering is a machine learning
technique. This technique generally allocates the dataset into `K` distinct clusters
based on their similarity. This also allows to minimize the variability within each
cluster, and then to maximize it between clusters. Thus, the grouping observations
where data points were assigned within a same cluster are generally more as they
differ with one another in their datapoints on the aspects of their contrasting clusters.
Hence, this method of K-Means clustering model is well trained for these datasets for
an inconsequential telecommunications company like ours which eventually had two
items and they are tenure and monthly.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
```


```python
# Load the dataset
df = pd.read_csv('Dataset(ATS).csv')
df.head()
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>Contract</th>
      <th>MonthlyCharges</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>1</td>
      <td>No</td>
      <td>No</td>
      <td>DSL</td>
      <td>Month-to-month</td>
      <td>29.85</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>34</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>One year</td>
      <td>56.95</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Month-to-month</td>
      <td>53.85</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>45</td>
      <td>No</td>
      <td>No</td>
      <td>DSL</td>
      <td>One year</td>
      <td>42.30</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>Month-to-month</td>
      <td>70.70</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.count()
```




    gender             7043
    SeniorCitizen      7043
    Dependents         7043
    tenure             7043
    PhoneService       7043
    MultipleLines      7043
    InternetService    7043
    Contract           7043
    MonthlyCharges     7043
    Churn              7043
    dtype: int64




```python
df.dtypes
```




    gender              object
    SeniorCitizen        int64
    Dependents          object
    tenure               int64
    PhoneService        object
    MultipleLines       object
    InternetService     object
    Contract            object
    MonthlyCharges     float64
    Churn               object
    dtype: object




```python
df.isnull().sum()
```




    gender             0
    SeniorCitizen      0
    Dependents         0
    tenure             0
    PhoneService       0
    MultipleLines      0
    InternetService    0
    Contract           0
    MonthlyCharges     0
    Churn              0
    dtype: int64




```python
df.head()
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>Contract</th>
      <th>MonthlyCharges</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>1</td>
      <td>No</td>
      <td>No</td>
      <td>DSL</td>
      <td>Month-to-month</td>
      <td>29.85</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>34</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>One year</td>
      <td>56.95</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Month-to-month</td>
      <td>53.85</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>45</td>
      <td>No</td>
      <td>No</td>
      <td>DSL</td>
      <td>One year</td>
      <td>42.30</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>Month-to-month</td>
      <td>70.70</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_clean = df.dropna()
```


```python
df_clean.head()
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>Contract</th>
      <th>MonthlyCharges</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>1</td>
      <td>No</td>
      <td>No</td>
      <td>DSL</td>
      <td>Month-to-month</td>
      <td>29.85</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>34</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>One year</td>
      <td>56.95</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Month-to-month</td>
      <td>53.85</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>45</td>
      <td>No</td>
      <td>No</td>
      <td>DSL</td>
      <td>One year</td>
      <td>42.30</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>Month-to-month</td>
      <td>70.70</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df_clean.columns)
```

    Index(['gender', 'SeniorCitizen', 'Dependents', 'tenure', 'PhoneService',
           'MultipleLines', 'InternetService', 'Contract', 'MonthlyCharges',
           'Churn'],
          dtype='object')
    


```python
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_features = encoder.fit_transform(df[['gender', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'Contract']])
df_encoded = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())
df_encoded = pd.concat([df[['SeniorCitizen', 'tenure', 'MonthlyCharges']], df_encoded], axis=1)
```


```python
df_encoded.head()
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SeniorCitizen</th>
      <th>tenure</th>
      <th>MonthlyCharges</th>
      <th>gender_Male</th>
      <th>Dependents_Yes</th>
      <th>PhoneService_Yes</th>
      <th>MultipleLines_Yes</th>
      <th>InternetService_Fiber optic</th>
      <th>Contract_One year</th>
      <th>Contract_Two year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>29.85</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>34</td>
      <td>56.95</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>53.85</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>45</td>
      <td>42.30</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>2</td>
      <td>70.70</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Scale the features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_encoded)
```


```python
# Define the range of cluster numbers to evaluate
cluster_range = range(1, 11)  # Testing from 1 to 10 clusters
wcss = []

# Compute WCSS for different numbers of clusters
for i in cluster_range:
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, wcss, marker='o', linestyle='-', color='b', label='WCSS')

# Find and mark the elbow point
# Compute differences and second differences to identify the elbow
diffs = np.diff(wcss)
second_diffs = np.diff(diffs)
elbow_index = np.argmin(second_diffs) + 1  # Add 1 to align with cluster numbers

# Plot the elbow point
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


    
![png](output_12_0.png)
    


    The optimal number of clusters is approximately: 6
    


```python
from sklearn.cluster import KMeans
import numpy as np

# Function to perform clustering and print cluster centers
def perform_clustering(n_clusters, df_scaled):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(df_scaled)
    cluster_centers = kmeans.cluster_centers_
    return cluster_labels, cluster_centers

# Perform clustering with 6 clusters
labels_6, centers_6 = perform_clustering(6, df_scaled)

# Print cluster centers for both cluster numbers
print("Cluster Centers for 6 Clusters:\n", centers_6)

# Example: Check the number of samples per cluster
unique, counts_6 = np.unique(labels_6, return_counts=True)

print("Cluster Counts for 6 Clusters:\n", dict(zip(unique, counts_6)))
```

    Cluster Centers for 6 Clusters:
     [[-0.02619242 -0.02580091 -0.75555595  0.01981342  0.00538018 -3.05401039
      -0.85417615 -0.88565976  0.00852265 -0.01074678]
     [-0.43991649 -0.48833165  0.7287298  -0.01883475 -0.20490532  0.32743831
       0.23376669  1.12239484 -0.51424938 -0.55518793]
     [-0.32913259  0.35333812 -0.03194613  0.02215468  0.18463774  0.32743831
       0.01990346 -0.1872091   1.94458183 -0.56297505]
     [-0.35752375 -0.84971883 -0.81281749  0.0273646  -0.03840524  0.32743831
      -0.53206706 -0.88565976 -0.51424938 -0.42766037]
     [ 2.27315869 -0.00714329  0.81552007 -0.04213582 -0.50913102  0.32743831
       0.52384566  0.99538869 -0.16022194 -0.36562528]
     [-0.32646418  1.08765403 -0.10223388 -0.00879241  0.42412771  0.32743831
       0.34440978 -0.3728906  -0.51424938  1.75435241]]
    Cluster Counts for 6 Clusters:
     {0: 682, 1: 1502, 2: 1200, 3: 1383, 4: 889, 5: 1387}
    


```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np

def plot_cluster_comparison(df_scaled, labels_6, centers_6):
    # Apply PCA to reduce dimensions to 2D
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df_scaled)
    
    # Transform the cluster centers to 2D space
    centers_6_pca = pca.transform(centers_6)
    
    plt.figure(figsize=(14, 7))
    
    # Plot 6 Clusters
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=labels_6, palette="tab10", alpha=0.6)
    plt.scatter(centers_6_pca[:, 0], centers_6_pca[:, 1], s=200, c='red', marker='x', label='Centroids (6 Clusters)')
    plt.title('6 Clusters')
    plt.legend()
    
    plt.show()

# Assuming you have the cluster labels and centers from previous steps
plot_cluster_comparison(df_scaled, labels_6, centers_6)
```


    
![png](output_14_0.png)
    



```python
from sklearn.cluster import KMeans

# Fit the KMeans model for 6 clusters
kmeans_6 = KMeans(n_clusters=6, init='k-means++', max_iter=300, n_init=10, random_state=0)
labels_6 = kmeans_6.fit_predict(df_scaled)

# Count the number of samples in each cluster for 6 clusters
cluster_counts_6 = np.bincount(labels_6)
print("Cluster Counts for 6 Clusters:")
for i, count in enumerate(cluster_counts_6):
    print(f"Cluster {i}: {count}")
```

    Cluster Counts for 6 Clusters:
    Cluster 0: 682
    Cluster 1: 1502
    Cluster 2: 1200
    Cluster 3: 1383
    Cluster 4: 889
    Cluster 5: 1387
    


```python
import pandas as pd
from sklearn.cluster import KMeans

# Assuming df is your original DataFrame and df_scaled is your scaled dataset

# Define the number of clusters (K)
n_clusters = 6  

# Initialize and fit K-Means model
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(df_scaled)  # df_scaled is your scaled dataset

# Get cluster labels for each data point
labels = kmeans.labels_

# Get cluster centers
centers = kmeans.cluster_centers_

# Optionally, create a DataFrame to analyze cluster characteristics
df_with_labels = df.copy()  # Copy original DataFrame
df_with_labels['Cluster'] = labels  # Add cluster labels

# Retrieve column names from the original DataFrame
columns = df.columns

# Create DataFrame for cluster centers using the column names
centers_df = pd.DataFrame(centers, columns=columns)
print("Cluster Centers:")
print(centers_df)

# Display cluster distribution
print("\nCluster Distribution:")
print(df_with_labels['Cluster'].value_counts())
```

    Cluster Centers:
         gender  SeniorCitizen  Dependents    tenure  PhoneService  MultipleLines  \
    0 -0.026192      -0.025801   -0.755556  0.019813      0.005380      -3.054010   
    1 -0.342209       0.350756   -0.033762  0.022314      0.187024       0.327438   
    2 -0.422587      -0.654906    0.019220  0.990532     -0.134979       0.327438   
    3 -0.428103      -0.640960    0.049459 -1.009559     -0.123303       0.327438   
    4  2.273159      -0.035164    0.770449 -0.037714     -0.511949       0.327438   
    5 -0.331989       0.985747   -0.171460 -0.006770      0.415047       0.327438   
    
       InternetService  Contract  MonthlyCharges     Churn  
    0        -0.854176 -0.885660        0.008523 -0.010747  
    1         0.017512 -0.185387        1.944582 -0.562975  
    2        -0.116936  0.173913       -0.514249 -0.557994  
    3        -0.090065  0.213833       -0.514249 -0.561277  
    4         0.483309  0.930248       -0.156892 -0.372689  
    5         0.277710 -0.392333       -0.514249  1.771464  
    
    Cluster Distribution:
    Cluster
    5    1458
    2    1409
    3    1378
    1    1194
    4     922
    0     682
    Name: count, dtype: int64
    


```python
# Save the clustered dataset
df_with_labels.head()
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>Contract</th>
      <th>MonthlyCharges</th>
      <th>Churn</th>
      <th>Cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>1</td>
      <td>No</td>
      <td>No</td>
      <td>DSL</td>
      <td>Month-to-month</td>
      <td>29.85</td>
      <td>No</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>34</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>One year</td>
      <td>56.95</td>
      <td>No</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Month-to-month</td>
      <td>53.85</td>
      <td>Yes</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>45</td>
      <td>No</td>
      <td>No</td>
      <td>DSL</td>
      <td>One year</td>
      <td>42.30</td>
      <td>No</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>Month-to-month</td>
      <td>70.70</td>
      <td>Yes</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


**3. Summary**
Import Libraries: - It generally allows in importing all the required libraries for data
manipulation, scaling and clustering and Visualisation.

Load Dataset: Load customer dataset.

Data preprocessing: It allows to rename the desired Column names and taking relevant features Tenure & Monthly respectively.
Normalize Data -This feature makes sure that the selected features are on a similar scale.

Elbow Method - Finding the best Number of Clusters with Python coding procedure.

Train Model: Train the K-Means model using the optimal number of clusters.
Interpret Results: Shows the centres of clusters.
