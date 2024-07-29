1. Introduction
The popular clustering process called K-Means clustering is a machine learning
technique. This technique generally allocates the dataset into `K` distinct clusters
based on their similarity. This also allows to minimize the variability within each
cluster, and then to maximize it between clusters. Thus, the grouping observations
where data points were assigned within a same cluster are generally more as they
differ with one another in their datapoints on the aspects of their contrasting clusters.
Hence, this method of K-Means clustering model is well trained for these datasets for
an inconsequential telecommunications company like ours which eventually had two
items and they are tenure and monthly.


2. Steps and Code
Step 1: Importing Libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
• Purpose: It works on importing the library functions for data manipulation,
scaling, clustering, visualization and many more features.

Step 2: Load the Dataset
df = pd.read_csv('Dataset(ATS).csv')
• Purpose: This feature works to load the dataset containing customer data.

Step 3: Data Preprocessing
df.columns = df.columns.str.lower().str.replace(' ', '_')
• Purpose: This feature works on converting the space between every column
name to underscores and again column names to lowercase.

Step 4: Select Relevant Features
df_segment = df[['tenure', 'monthly charges']]
• Purpose: This another feature works on selecting the tenure and monthly
charges to do cluster analysis.

Step 5: Normalizing the given dataset.
scaler = StandardScaler()
df_normalized = scaler.fit_transform(df_segment)
df_normalized = pd.DataFrame(df_normalized, columns=df_segment.columns)
• Purpose: This different feature also works on normalizing the data on a similar
scale and thus improving the performance.

Step 6: Determine the Optimal Number of Clusters Using the Elbow Method
sse = []
for k in range(1, 11):
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(df_normalized)
sse.append(kmeans.inertia_)
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

Purpose: This feature works basically to plot the sum of squared errors (SSE)
for various values of k to find the ideal number of clusters using the Elbow
Method.

Step 7: Train the K-Means Model
optimal_clusters = 3 #Based on the elbow plot
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(df_normalized)

Purpose: This feature works totain the K-Means clustering model assuming
clusters (k=3).

Step 8: Interpret the Results
cluster_centers = kmeans.cluster_centers_
cluster_centers = scaler.inverse_transform(cluster_centers)
cluster_centers_df = pd.DataFrame(cluster_centers, columns=df_segment.columns)
print("Cluster Centers:")
print(cluster_centers_df)

Purpose: This works on comprehending the average values for each cluster,
locate and show the cluster centres.

Step 9: Visualize the Clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['tenure'], df['monthlycharges'], c=df['cluster'], cmap='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=300, c='red')
plt.title('Customer Segments')
plt.xlabel('Tenure')
plt.ylabel('Monthly Charges')
plt.show()

Purpose: This important feature works on visualizing the clusters and their
centres


3. Summary
Import Libraries: - It generally allows in importing all the required libraries for data
manipulation, scaling and clustering and Visualisation.

Load Dataset: Load customer dataset.

Data preprocessing: It allows to rename the desired Column names and taking relevant features Tenure & Monthly respectively.
Normalize Data -This feature makes sure that the selected features are on a similar scale.

Elbow Method - Finding the best Number of Clusters with Python coding procedure.

Train Model: Train the K-Means model using the optimal number of clusters.
Interpret Results: Shows the centres of clusters.