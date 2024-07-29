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


**2. clustering_analysis.ipynb**

**3. Summary**
Import Libraries: - It generally allows in importing all the required libraries for data
manipulation, scaling and clustering and Visualisation.

Load Dataset: Load customer dataset.

Data preprocessing: It allows to rename the desired Column names and taking relevant features Tenure & Monthly respectively.
Normalize Data -This feature makes sure that the selected features are on a similar scale.

Elbow Method - Finding the best Number of Clusters with Python coding procedure.

Train Model: Train the K-Means model using the optimal number of clusters.
Interpret Results: Shows the centres of clusters.
