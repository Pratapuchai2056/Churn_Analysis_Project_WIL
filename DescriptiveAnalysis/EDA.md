```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
```


```python
df = pd.read_csv('Dataset(ATS).csv')
```


```python
#1 Data Exploration after inspection
df.head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
    <tr>
      <th>5</th>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>8</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>Month-to-month</td>
      <td>99.65</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>22</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>Month-to-month</td>
      <td>89.10</td>
      <td>No</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>10</td>
      <td>No</td>
      <td>No</td>
      <td>DSL</td>
      <td>Month-to-month</td>
      <td>29.75</td>
      <td>No</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>28</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>Month-to-month</td>
      <td>104.80</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>62</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>One year</td>
      <td>56.15</td>
      <td>No</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>13</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Month-to-month</td>
      <td>49.95</td>
      <td>No</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>16</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Two year</td>
      <td>18.95</td>
      <td>No</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>58</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>One year</td>
      <td>100.35</td>
      <td>No</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>49</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>Month-to-month</td>
      <td>103.70</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>25</td>
      <td>Yes</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>Month-to-month</td>
      <td>105.50</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df.head(10))
```

       gender  SeniorCitizen Dependents  tenure PhoneService MultipleLines  \
    0  Female              0         No       1           No            No   
    1    Male              0         No      34          Yes            No   
    2    Male              0         No       2          Yes            No   
    3    Male              0         No      45           No            No   
    4  Female              0         No       2          Yes            No   
    5  Female              0         No       8          Yes           Yes   
    6    Male              0        Yes      22          Yes           Yes   
    7  Female              0         No      10           No            No   
    8  Female              0         No      28          Yes           Yes   
    9    Male              0        Yes      62          Yes            No   
    
      InternetService        Contract  MonthlyCharges Churn  
    0             DSL  Month-to-month           29.85    No  
    1             DSL        One year           56.95    No  
    2             DSL  Month-to-month           53.85   Yes  
    3             DSL        One year           42.30    No  
    4     Fiber optic  Month-to-month           70.70   Yes  
    5     Fiber optic  Month-to-month           99.65   Yes  
    6     Fiber optic  Month-to-month           89.10    No  
    7             DSL  Month-to-month           29.75    No  
    8     Fiber optic  Month-to-month          104.80   Yes  
    9             DSL        One year           56.15    No  
    


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
## Data Cleaning
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
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.columns
```




    Index(['gender', 'seniorcitizen', 'dependents', 'tenure', 'phoneservice',
           'multiplelines', 'internetservice', 'contract', 'monthlycharges',
           'churn'],
          dtype='object')




```python
fig = px.pie(df, names = 'churn', template = 'simple_white', title = 'Churn')
fig.update_traces(rotation = 90, pull = [0.1], textinfo = 'percent+label')
fig.show()
```


<div>                            <div id="abd10966-f9fd-47ac-a9e9-0d7cb7de6d77" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("abd10966-f9fd-47ac-a9e9-0d7cb7de6d77")) {                    Plotly.newPlot(                        "abd10966-f9fd-47ac-a9e9-0d7cb7de6d77",                        [{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"hovertemplate":"churn=%{label}\u003cextra\u003e\u003c\u002fextra\u003e","labels":["No","No","Yes","No","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","Yes","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","Yes","No","Yes","No","No","Yes","Yes","No","No","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","Yes","No","Yes","Yes","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","Yes","No","No","Yes","Yes","No","No","Yes","Yes","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","No","No","No","Yes","No","No","Yes","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","Yes","No","No","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","No","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","Yes","Yes","No","No","No","Yes","Yes","No","Yes","No","Yes","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","Yes","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","Yes","No","No","Yes","Yes","No","No","No","Yes","No","No","Yes","No","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","No","Yes","No","No","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","No","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","Yes","Yes","No","No","No","Yes","Yes","No","Yes","No","No","No","Yes","Yes","Yes","No","Yes","No","Yes","No","No","Yes","Yes","No","No","Yes","Yes","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","Yes","No","Yes","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","Yes","No","No","Yes","No","No","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","Yes","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","Yes","Yes","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","Yes","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","Yes","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","Yes","No","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","Yes","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","Yes","Yes","No","No","No","Yes","No","Yes","No","Yes","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","Yes","No","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","No","Yes","Yes","No","No","No","Yes","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","Yes","Yes","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","Yes","Yes","No","Yes","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","Yes","No","Yes","No","Yes","No","No","No","Yes","No","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","No","No","Yes","No","No","No","Yes","No","Yes","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","Yes","No","No","No","Yes","No","No","Yes","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","No","No","Yes","No","No","Yes","Yes","No","No","Yes","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","No","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","No","Yes","No","Yes","No","No","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","Yes","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","Yes","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","Yes","Yes","No","Yes","Yes","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","Yes","No","No","No","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","Yes","No","Yes","No","No","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","Yes","No","No","No","No","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","Yes","Yes","No","No","No","No","Yes","Yes","No","Yes","No","Yes","No","No","Yes","No","Yes","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","Yes","No","No","Yes","No","No","Yes","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","Yes","No","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","No","No","No","No","No","Yes","No","Yes","Yes","No","No","Yes","No","No","No","Yes","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","No","Yes","No","Yes","No","Yes","No","Yes","No","No","Yes","Yes","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","Yes","Yes","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","No","Yes","No","Yes","No","Yes","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","Yes","No","No","Yes","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","Yes","No","No","Yes","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","No","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","No","No","Yes","No","Yes","Yes","No","No","No","Yes","No","Yes","Yes","No","No","No","Yes","Yes","No","Yes","No","Yes","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","Yes","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","No","No","Yes","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","No"],"legendgroup":"","name":"","showlegend":true,"type":"pie","pull":[0.1],"rotation":90,"textinfo":"percent+label"}],                        {"template":{"data":{"barpolar":[{"marker":{"line":{"color":"white","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"rgb(36,36,36)"},"error_y":{"color":"rgb(36,36,36)"},"marker":{"line":{"color":"white","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"rgb(36,36,36)","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"rgb(36,36,36)"},"baxis":{"endlinecolor":"rgb(36,36,36)","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"rgb(36,36,36)"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"contour"}],"heatmapgl":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"heatmapgl"}],"heatmap":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"histogram2d"}],"histogram":[{"marker":{"line":{"color":"white","width":0.6}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scattermapbox"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scatterpolar"}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"rgb(237,237,237)"},"line":{"color":"white"}},"header":{"fill":{"color":"rgb(217,217,217)"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"colorscale":{"diverging":[[0.0,"rgb(103,0,31)"],[0.1,"rgb(178,24,43)"],[0.2,"rgb(214,96,77)"],[0.3,"rgb(244,165,130)"],[0.4,"rgb(253,219,199)"],[0.5,"rgb(247,247,247)"],[0.6,"rgb(209,229,240)"],[0.7,"rgb(146,197,222)"],[0.8,"rgb(67,147,195)"],[0.9,"rgb(33,102,172)"],[1.0,"rgb(5,48,97)"]],"sequential":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"sequentialminus":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]]},"colorway":["#1F77B4","#FF7F0E","#2CA02C","#D62728","#9467BD","#8C564B","#E377C2","#7F7F7F","#BCBD22","#17BECF"],"font":{"color":"rgb(36,36,36)"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"white","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"white","polar":{"angularaxis":{"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside"},"bgcolor":"white","radialaxis":{"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside"}},"scene":{"xaxis":{"backgroundcolor":"white","gridcolor":"rgb(232,232,232)","gridwidth":2,"linecolor":"rgb(36,36,36)","showbackground":true,"showgrid":false,"showline":true,"ticks":"outside","zeroline":false,"zerolinecolor":"rgb(36,36,36)"},"yaxis":{"backgroundcolor":"white","gridcolor":"rgb(232,232,232)","gridwidth":2,"linecolor":"rgb(36,36,36)","showbackground":true,"showgrid":false,"showline":true,"ticks":"outside","zeroline":false,"zerolinecolor":"rgb(36,36,36)"},"zaxis":{"backgroundcolor":"white","gridcolor":"rgb(232,232,232)","gridwidth":2,"linecolor":"rgb(36,36,36)","showbackground":true,"showgrid":false,"showline":true,"ticks":"outside","zeroline":false,"zerolinecolor":"rgb(36,36,36)"}},"shapedefaults":{"fillcolor":"black","line":{"width":0},"opacity":0.3},"ternary":{"aaxis":{"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside"},"baxis":{"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside"},"bgcolor":"white","caxis":{"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside"}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside","title":{"standoff":15},"zeroline":false,"zerolinecolor":"rgb(36,36,36)"},"yaxis":{"automargin":true,"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside","title":{"standoff":15},"zeroline":false,"zerolinecolor":"rgb(36,36,36)"}}},"legend":{"tracegroupgap":0},"title":{"text":"Churn"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('abd10966-f9fd-47ac-a9e9-0d7cb7de6d77');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
fig = px.pie(df, names = 'gender', template = 'simple_white', title = 'Gender')
fig.update_traces(rotation = 90, pull = [0.1], textinfo = 'percent+label')
fig.show()
```


<div>                            <div id="2dc69def-7fce-41b9-a10d-f42a5f35a8e6" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("2dc69def-7fce-41b9-a10d-f42a5f35a8e6")) {                    Plotly.newPlot(                        "2dc69def-7fce-41b9-a10d-f42a5f35a8e6",                        [{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"hovertemplate":"gender=%{label}\u003cextra\u003e\u003c\u002fextra\u003e","labels":["Female","Male","Male","Male","Female","Female","Male","Female","Female","Male","Male","Male","Male","Male","Male","Female","Female","Male","Female","Female","Male","Male","Male","Female","Male","Female","Male","Male","Male","Female","Female","Male","Female","Male","Male","Female","Male","Female","Male","Female","Male","Female","Female","Female","Female","Female","Male","Female","Male","Female","Female","Female","Female","Female","Female","Male","Female","Male","Female","Female","Female","Male","Male","Female","Female","Female","Male","Female","Female","Male","Male","Female","Male","Male","Female","Female","Female","Female","Male","Female","Female","Female","Female","Female","Female","Male","Female","Female","Female","Male","Male","Male","Male","Female","Male","Female","Male","Male","Male","Female","Male","Female","Male","Female","Male","Male","Female","Female","Male","Male","Male","Female","Female","Female","Female","Male","Female","Female","Female","Male","Female","Female","Male","Female","Female","Male","Female","Male","Male","Male","Female","Female","Male","Male","Male","Female","Female","Female","Male","Female","Female","Female","Female","Male","Male","Female","Male","Male","Male","Male","Female","Male","Female","Male","Female","Female","Female","Male","Male","Male","Female","Female","Female","Male","Female","Male","Female","Male","Male","Female","Male","Female","Male","Female","Female","Female","Female","Female","Female","Male","Female","Female","Male","Male","Female","Female","Male","Male","Female","Female","Male","Male","Female","Female","Male","Female","Male","Male","Male","Male","Female","Female","Male","Male","Male","Female","Male","Male","Female","Male","Male","Female","Male","Female","Male","Female","Male","Male","Male","Female","Female","Female","Male","Male","Female","Male","Female","Female","Male","Female","Male","Male","Male","Female","Female","Male","Female","Male","Female","Male","Female","Male","Female","Female","Female","Male","Male","Male","Male","Male","Female","Female","Male","Male","Male","Female","Female","Male","Female","Female","Male","Male","Male","Female","Female","Female","Male","Female","Male","Female","Female","Male","Male","Male","Male","Male","Female","Male","Male","Female","Female","Female","Female","Female","Male","Female","Female","Male","Male","Male","Male","Male","Male","Female","Female","Male","Female","Male","Male","Female","Male","Female","Male","Female","Male","Male","Male","Male","Male","Male","Female","Female","Female","Female","Male","Male","Male","Female","Male","Male","Female","Female","Male","Female","Male","Female","Female","Female","Female","Female","Female","Female","Male","Female","Female","Male","Female","Female","Female","Male","Female","Female","Female","Male","Female","Female","Female","Female","Female","Female","Male","Male","Female","Male","Female","Female","Female","Male","Male","Male","Male","Female","Male","Female","Male","Male","Male","Female","Male","Male","Male","Male","Male","Female","Male","Female","Male","Female","Female","Male","Female","Male","Male","Male","Male","Male","Female","Female","Male","Male","Female","Male","Female","Female","Female","Female","Male","Female","Male","Female","Female","Female","Male","Female","Male","Male","Male","Male","Female","Male","Male","Female","Female","Male","Female","Female","Male","Female","Female","Female","Female","Female","Male","Male","Female","Female","Male","Female","Male","Female","Female","Female","Male","Male","Male","Female","Male","Male","Male","Male","Female","Female","Female","Female","Female","Female","Female","Female","Male","Male","Female","Female","Male","Female","Female","Female","Male","Female","Male","Male","Male","Female","Male","Male","Male","Male","Male","Male","Female","Female","Female","Male","Male","Male","Female","Male","Male","Female","Male","Female","Male","Female","Female","Female","Female","Male","Female","Male","Female","Male","Male","Female","Female","Male","Male","Male","Male","Female","Female","Female","Male","Female","Female","Female","Male","Female","Male","Male","Female","Female","Male","Male","Female","Female","Male","Male","Female","Female","Female","Male","Male","Male","Male","Female","Male","Male","Male","Male","Male","Female","Female","Male","Male","Male","Female","Male","Female","Female","Female","Female","Female","Female","Female","Male","Male","Male","Male","Female","Male","Male","Male","Female","Male","Female","Male","Female","Female","Male","Male","Male","Male","Male","Male","Male","Male","Male","Male","Male","Female","Female","Female","Male","Female","Male","Female","Male","Male","Male","Male","Male","Female","Female","Female","Female","Female","Female","Male","Male","Male","Male","Male","Male","Male","Male","Male","Female","Male","Male","Male","Male","Male","Male","Female","Male","Female","Male","Male","Male","Male","Female","Male","Female","Male","Female","Male","Female","Male","Female","Female","Female","Male","Female","Male","Female","Female","Female","Male","Female","Female","Male","Male","Female","Male","Male","Female","Female","Male","Female","Male","Male","Male","Female","Male","Female","Female","Male","Male","Male","Female","Male","Female","Female","Male","Female","Female","Male","Male","Male","Male","Male","Female","Female","Male","Male","Female","Female","Female","Male","Female","Female","Female","Male","Female","Male","Male","Male","Female","Female","Male","Male","Male","Female","Male","Female","Female","Female","Male","Male","Male","Male","Male","Male","Male","Female","Female","Female","Male","Male","Male","Male","Male","Male","Male","Female","Male","Female","Female","Male","Male","Male","Male","Female","Female","Female","Male","Male","Female","Male","Female","Male","Male","Male","Female","Male","Male","Male","Female","Female","Female","Male","Female","Male","Male","Female","Female","Male","Female","Male","Male","Female","Female","Female","Male","Female","Female","Female","Male","Female","Male","Female","Female","Male","Male","Male","Female","Male","Male","Female","Female","Female","Male","Female","Male","Female","Male","Female","Male","Male","Male","Male","Male","Male","Female","Female","Male","Female","Female","Male","Female","Male","Male","Female","Male","Female","Male","Female","Male","Female","Female","Male","Female","Male","Female","Female","Female","Male","Male","Female","Female","Male","Female","Female","Male","Female","Male","Female","Male","Male","Male","Male","Male","Male","Female","Male","Male","Female","Female","Female","Male","Male","Female","Male","Male","Male","Female","Female","Male","Female","Male","Male","Female","Male","Female","Female","Female","Male","Male","Male","Male","Male","Female","Female","Female","Female","Male","Female","Female","Female","Male","Female","Female","Male","Male","Female","Female","Female","Male","Male","Female","Male","Male","Male","Male","Female","Male","Male","Male","Female","Male","Male","Female","Male","Male","Female","Male","Female","Female","Female","Female","Male","Male","Male","Female","Female","Female","Male","Female","Female","Male","Female","Male","Male","Male","Female","Female","Male","Male","Female","Female","Male","Female","Female","Male","Female","Male","Male","Male","Male","Female","Female","Male","Female","Male","Male","Male","Male","Female","Male","Female","Female","Male","Male","Male","Female","Female","Female","Female","Female","Female","Female","Female","Male","Male","Female","Female","Female","Female","Female","Female","Female","Female","Female","Male","Female","Female","Female","Male","Female","Male","Male","Male","Female","Male","Female","Male","Male","Female","Male","Male","Female","Male","Male","Male","Female","Male","Male","Male","Male","Female","Female","Female","Male","Male","Male","Female","Male","Male","Female","Male","Female","Female","Male","Female","Male","Female","Female","Female","Male","Male","Male","Female","Female","Female","Female","Male","Female","Female","Female","Male","Male","Female","Male","Male","Female","Male","Female","Female","Female","Male","Male","Male","Female","Male","Female","Male","Male","Male","Female","Male","Female","Female","Male","Female","Female","Male","Female","Male","Female","Male","Female","Female","Male","Female","Female","Female","Male","Male","Female","Male","Male","Male","Female","Female","Male","Female","Female","Female","Male","Male","Female","Male","Female","Female","Female","Male","Female","Female","Female","Male","Male","Male","Female","Female","Male","Female","Female","Male","Male","Male","Male","Female","Female","Male","Male","Male","Female","Female","Female","Female","Female","Male","Male","Male","Male","Female","Female","Female","Male","Male","Female","Female","Male","Male","Female","Male","Male","Female","Male","Male","Male","Male","Female","Male","Male","Male","Female","Female","Female","Female","Male","Male","Female","Female","Female","Male","Male","Female","Male","Female","Male","Male","Female","Female","Male","Male","Male","Male","Female","Male","Male","Male","Female","Female","Male","Female","Female","Female","Male","Male","Male","Female","Male","Male","Male","Male","Female","Male","Female","Female","Male","Female","Female","Male","Female","Female","Male","Female","Female","Female","Female","Female","Male","Female","Female","Female","Male","Female","Male","Male","Male","Male","Male","Female","Male","Female","Female","Female","Female","Male","Male","Male","Male","Male","Female","Male","Female","Female","Female","Female","Male","Male","Male","Male","Female","Female","Female","Male","Male","Female","Female","Male","Female","Male","Female","Male","Male","Female","Female","Female","Male","Male","Female","Male","Female","Male","Female","Male","Male","Male","Female","Male","Male","Male","Female","Male","Female","Female","Male","Female","Male","Male","Male","Male","Female","Male","Male","Male","Male","Female","Female","Male","Male","Female","Male","Female","Male","Female","Female","Female","Male","Female","Female","Male","Female","Female","Female","Female","Female","Female","Female","Male","Female","Female","Female","Male","Male","Female","Male","Male","Male","Male","Female","Male","Male","Female","Female","Female","Female","Male","Male","Female","Female","Male","Female","Male","Male","Female","Female","Male","Male","Male","Male","Female","Male","Male","Female","Female","Female","Female","Female","Male","Female","Female","Female","Male","Female","Female","Female","Male","Female","Female","Male","Male","Male","Female","Female","Male","Female","Male","Male","Male","Male","Female","Female","Female","Female","Female","Male","Female","Female","Male","Female","Male","Male","Female","Male","Female","Male","Male","Male","Female","Male","Female","Male","Female","Female","Female","Male","Male","Female","Female","Male","Male","Male","Female","Female","Female","Male","Male","Female","Female","Male","Male","Female","Female","Male","Male","Male","Male","Female","Female","Female","Male","Male","Male","Female","Male","Male","Female","Female","Male","Male","Female","Female","Female","Female","Male","Male","Female","Female","Female","Male","Male","Female","Female","Male","Female","Female","Female","Female","Male","Female","Female","Male","Female","Female","Male","Male","Female","Female","Female","Male","Female","Male","Male","Male","Female","Female","Male","Male","Female","Male","Female","Male","Female","Male","Male","Male","Male","Male","Female","Female","Male","Male","Male","Female","Female","Male","Female","Female","Male","Female","Female","Male","Male","Male","Male","Male","Male","Female","Male","Male","Male","Male","Male","Female","Female","Male","Male","Female","Male","Male","Male","Female","Male","Male","Male","Female","Male","Male","Female","Male","Male","Female","Male","Female","Male","Male","Male","Female","Male","Male","Female","Female","Female","Male","Female","Female","Male","Female","Female","Female","Male","Male","Female","Female","Male","Female","Male","Male","Female","Male","Male","Female","Female","Male","Male","Male","Female","Male","Male","Female","Female","Female","Female","Female","Female","Male","Female","Female","Female","Male","Female","Male","Female","Female","Female","Male","Female","Female","Male","Female","Male","Female","Female","Male","Male","Female","Male","Female","Male","Male","Female","Female","Male","Male","Female","Female","Male","Male","Male","Male","Female","Female","Male","Female","Female","Male","Female","Female","Male","Male","Female","Male","Male","Female","Male","Female","Male","Female","Female","Male","Female","Male","Female","Male","Female","Female","Female","Male","Female","Male","Female","Female","Female","Male","Female","Male","Female","Female","Female","Male","Female","Male","Female","Male","Female","Male","Male","Female","Male","Male","Male","Female","Male","Male","Male","Male","Male","Male","Male","Female","Male","Male","Male","Female","Male","Male","Female","Male","Female","Female","Female","Female","Female","Female","Male","Male","Female","Male","Male","Female","Male","Female","Male","Female","Female","Male","Female","Male","Female","Male","Female","Male","Female","Female","Female","Male","Male","Female","Female","Male","Male","Male","Male","Female","Female","Male","Male","Male","Male","Female","Male","Male","Female","Female","Female","Male","Female","Female","Male","Male","Male","Female","Female","Female","Female","Male","Female","Male","Male","Female","Male","Male","Female","Male","Female","Male","Female","Male","Male","Male","Male","Male","Female","Male","Male","Female","Male","Female","Male","Female","Female","Male","Male","Female","Male","Female","Male","Male","Female","Female","Female","Male","Male","Female","Male","Male","Female","Male","Female","Female","Female","Female","Female","Female","Male","Male","Female","Male","Male","Female","Male","Male","Male","Male","Female","Female","Female","Male","Male","Male","Female","Male","Female","Female","Female","Female","Female","Male","Female","Male","Male","Female","Male","Female","Female","Male","Female","Male","Male","Male","Male","Female","Male","Female","Female","Female","Female","Male","Female","Male","Male","Female","Male","Male","Male","Female","Male","Female","Female","Male","Female","Male","Female","Male","Female","Female","Female","Male","Female","Female","Male","Male","Female","Female","Female","Female","Male","Female","Male","Male","Male","Female","Female","Male","Male","Female","Female","Male","Female","Female","Male","Male","Male","Female","Male","Male","Male","Female","Female","Male","Female","Male","Male","Female","Male","Female","Male","Male","Male","Female","Male","Male","Male","Female","Male","Male","Male","Female","Male","Female","Male","Female","Male","Male","Male","Female","Female","Male","Female","Female","Male","Female","Male","Male","Female","Female","Female","Male","Male","Female","Female","Male","Female","Male","Female","Female","Male","Male","Male","Female","Female","Female","Female","Male","Male","Male","Male","Male","Male","Male","Male","Female","Male","Male","Male","Female","Female","Male","Female","Female","Male","Male","Female","Male","Male","Male","Female","Male","Male","Female","Male","Male","Male","Female","Male","Female","Male","Male","Female","Female","Male","Female","Female","Male","Female","Female","Male","Male","Male","Male","Male","Male","Female","Female","Male","Male","Female","Male","Male","Female","Male","Male","Female","Female","Female","Female","Female","Male","Male","Male","Male","Male","Male","Female","Male","Female","Female","Male","Male","Male","Female","Male","Male","Female","Female","Female","Male","Female","Female","Female","Male","Male","Male","Male","Male","Male","Female","Female","Female","Female","Female","Male","Male","Male","Male","Male","Female","Female","Male","Female","Male","Male","Female","Female","Female","Female","Male","Female","Female","Male","Female","Female","Male","Female","Male","Female","Female","Male","Female","Female","Male","Female","Male","Female","Male","Male","Female","Male","Male","Male","Female","Female","Male","Female","Female","Male","Female","Male","Male","Male","Female","Female","Male","Female","Male","Female","Female","Male","Male","Female","Female","Male","Male","Female","Male","Female","Female","Male","Male","Male","Female","Male","Male","Female","Female","Male","Male","Female","Female","Female","Female","Male","Female","Male","Male","Female","Male","Male","Male","Male","Female","Male","Female","Female","Male","Male","Male","Female","Female","Female","Male","Male","Female","Male","Female","Female","Male","Male","Male","Female","Male","Female","Male","Female","Male","Female","Female","Female","Female","Female","Male","Male","Female","Male","Female","Male","Male","Female","Male","Female","Male","Male","Female","Female","Male","Male","Female","Male","Female","Male","Female","Male","Male","Female","Male","Female","Female","Male","Female","Male","Female","Male","Male","Female","Male","Male","Male","Male","Male","Male","Male","Female","Female","Female","Male","Male","Female","Female","Female","Male","Female","Male","Male","Male","Female","Male","Male","Female","Male","Male","Female","Female","Female","Female","Female","Male","Female","Female","Female","Female","Male","Male","Male","Female","Female","Male","Male","Female","Female","Female","Female","Male","Female","Female","Male","Male","Male","Female","Male","Female","Male","Male","Female","Male","Male","Male","Female","Male","Male","Female","Male","Male","Male","Female","Female","Male","Female","Male","Female","Female","Male","Male","Male","Male","Male","Female","Female","Male","Female","Female","Male","Female","Female","Female","Female","Female","Female","Male","Male","Female","Male","Female","Male","Male","Female","Male","Female","Male","Male","Male","Male","Male","Male","Female","Male","Male","Female","Female","Female","Female","Male","Female","Female","Female","Female","Female","Male","Female","Female","Male","Female","Male","Male","Female","Female","Female","Male","Female","Female","Female","Female","Female","Male","Male","Male","Female","Female","Female","Female","Female","Male","Male","Male","Male","Female","Male","Female","Male","Female","Female","Male","Male","Male","Male","Male","Male","Female","Female","Female","Female","Female","Male","Male","Male","Male","Female","Male","Female","Female","Female","Male","Male","Male","Male","Female","Female","Male","Male","Male","Female","Male","Male","Female","Male","Male","Female","Female","Male","Male","Female","Female","Female","Female","Female","Female","Female","Female","Female","Male","Female","Female","Female","Male","Male","Male","Female","Male","Female","Female","Female","Female","Female","Female","Female","Female","Male","Male","Male","Female","Female","Male","Male","Male","Male","Male","Female","Female","Male","Male","Female","Male","Male","Male","Male","Female","Female","Female","Male","Male","Female","Male","Female","Male","Female","Male","Male","Female","Male","Male","Male","Female","Male","Male","Female","Female","Male","Female","Male","Female","Male","Male","Female","Male","Female","Female","Male","Female","Female","Female","Male","Male","Female","Female","Female","Female","Female","Female","Male","Female","Male","Female","Male","Female","Female","Male","Male","Female","Female","Male","Male","Male","Female","Female","Male","Female","Male","Male","Male","Male","Male","Female","Male","Female","Female","Female","Female","Female","Male","Male","Female","Male","Female","Female","Female","Female","Male","Female","Male","Female","Male","Male","Female","Female","Female","Male","Male","Male","Male","Female","Female","Male","Male","Female","Male","Male","Male","Male","Female","Female","Female","Male","Female","Female","Female","Male","Male","Male","Male","Male","Male","Female","Male","Male","Female","Female","Male","Male","Female","Male","Female","Male","Female","Male","Female","Male","Female","Male","Male","Male","Male","Female","Male","Male","Female","Female","Male","Female","Female","Male","Male","Female","Male","Female","Female","Female","Male","Male","Male","Female","Male","Male","Female","Male","Male","Male","Female","Female","Male","Female","Female","Female","Male","Male","Female","Male","Male","Male","Female","Female","Female","Male","Male","Female","Female","Female","Female","Male","Male","Female","Female","Female","Male","Female","Female","Male","Male","Female","Male","Male","Female","Male","Male","Male","Female","Female","Male","Male","Female","Male","Female","Female","Female","Male","Female","Female","Female","Female","Male","Female","Male","Female","Female","Female","Female","Female","Male","Female","Male","Female","Female","Female","Female","Female","Female","Female","Male","Female","Female","Female","Female","Female","Female","Female","Female","Female","Male","Male","Male","Female","Male","Female","Female","Female","Female","Male","Female","Male","Female","Male","Male","Male","Female","Male","Male","Female","Female","Female","Male","Male","Male","Female","Female","Male","Male","Female","Female","Male","Male","Male","Female","Female","Female","Male","Female","Male","Female","Male","Female","Female","Male","Male","Female","Male","Female","Female","Female","Female","Female","Male","Female","Male","Female","Male","Male","Male","Female","Male","Female","Female","Male","Male","Male","Female","Male","Female","Female","Male","Male","Male","Male","Male","Female","Male","Female","Male","Male","Male","Female","Male","Female","Female","Male","Male","Male","Female","Female","Male","Female","Female","Female","Male","Female","Female","Female","Male","Male","Female","Male","Female","Female","Male","Male","Female","Male","Female","Female","Male","Female","Male","Male","Male","Male","Male","Female","Male","Male","Female","Female","Male","Female","Female","Female","Male","Male","Female","Male","Female","Female","Female","Male","Male","Male","Female","Male","Female","Male","Male","Male","Male","Female","Male","Female","Male","Male","Female","Female","Male","Male","Male","Female","Male","Male","Male","Male","Female","Female","Female","Female","Male","Female","Female","Male","Male","Male","Male","Male","Female","Male","Female","Male","Male","Female","Female","Female","Male","Male","Male","Female","Female","Male","Female","Male","Female","Female","Female","Male","Male","Male","Female","Female","Male","Female","Male","Female","Male","Male","Male","Male","Female","Female","Male","Male","Female","Male","Female","Female","Male","Female","Male","Male","Female","Male","Male","Male","Female","Male","Male","Male","Male","Female","Male","Female","Male","Female","Male","Female","Female","Female","Female","Male","Female","Female","Female","Female","Female","Female","Male","Female","Male","Female","Male","Female","Male","Male","Male","Female","Female","Male","Male","Female","Male","Male","Male","Female","Female","Female","Male","Female","Male","Male","Female","Male","Female","Male","Male","Female","Male","Male","Male","Female","Female","Male","Female","Male","Female","Male","Female","Male","Female","Female","Male","Male","Female","Female","Male","Male","Female","Male","Female","Female","Female","Male","Male","Female","Female","Male","Male","Female","Female","Female","Male","Male","Male","Female","Male","Male","Male","Male","Female","Male","Female","Male","Male","Female","Female","Male","Male","Male","Male","Female","Male","Male","Female","Male","Male","Female","Male","Male","Male","Male","Male","Male","Male","Male","Female","Male","Male","Female","Male","Male","Female","Female","Female","Male","Male","Female","Female","Male","Male","Male","Female","Female","Male","Male","Female","Male","Male","Female","Male","Female","Male","Female","Male","Male","Male","Male","Male","Male","Female","Male","Male","Male","Female","Female","Female","Female","Male","Male","Male","Female","Male","Male","Male","Female","Female","Male","Male","Male","Male","Male","Male","Male","Male","Female","Male","Female","Female","Male","Male","Female","Male","Female","Male","Male","Female","Female","Male","Female","Male","Male","Female","Male","Male","Male","Female","Female","Male","Male","Male","Female","Male","Female","Male","Female","Male","Male","Female","Male","Male","Male","Male","Female","Female","Female","Female","Female","Female","Female","Female","Female","Male","Male","Female","Male","Male","Male","Female","Female","Male","Male","Male","Female","Female","Male","Female","Male","Male","Female","Female","Male","Female","Female","Female","Male","Female","Male","Female","Female","Male","Male","Male","Male","Male","Male","Male","Female","Male","Female","Male","Female","Female","Male","Female","Male","Female","Male","Male","Female","Female","Female","Female","Female","Male","Male","Female","Female","Female","Male","Female","Male","Male","Male","Male","Female","Female","Female","Male","Female","Female","Female","Male","Male","Male","Male","Female","Female","Male","Male","Male","Female","Male","Male","Female","Female","Female","Female","Female","Male","Male","Male","Male","Male","Male","Female","Female","Female","Female","Female","Male","Male","Female","Female","Female","Male","Female","Male","Male","Male","Female","Male","Male","Male","Male","Male","Male","Female","Female","Male","Male","Female","Male","Female","Male","Male","Female","Male","Male","Female","Female","Female","Female","Male","Female","Female","Female","Male","Female","Female","Male","Male","Male","Female","Female","Female","Female","Female","Male","Female","Male","Female","Male","Female","Male","Female","Female","Female","Male","Female","Female","Female","Male","Female","Male","Male","Female","Female","Male","Female","Male","Male","Male","Male","Female","Female","Male","Male","Male","Female","Male","Female","Male","Male","Female","Female","Male","Male","Female","Male","Female","Male","Male","Female","Male","Female","Male","Female","Male","Female","Male","Female","Male","Female","Male","Female","Male","Male","Male","Female","Male","Female","Female","Female","Female","Male","Female","Male","Male","Male","Female","Male","Male","Male","Female","Female","Female","Male","Female","Male","Male","Male","Male","Female","Male","Male","Male","Male","Female","Female","Female","Female","Female","Female","Female","Female","Female","Female","Female","Male","Female","Female","Male","Female","Male","Male","Female","Female","Female","Female","Female","Female","Male","Male","Female","Male","Female","Male","Male","Female","Male","Female","Female","Female","Female","Male","Male","Male","Female","Female","Male","Male","Male","Male","Female","Male","Female","Female","Female","Female","Female","Female","Male","Female","Male","Female","Female","Male","Female","Female","Male","Female","Female","Male","Female","Female","Female","Female","Male","Male","Male","Female","Female","Male","Female","Male","Male","Male","Female","Female","Female","Female","Male","Female","Female","Female","Female","Male","Male","Female","Male","Female","Male","Male","Female","Male","Female","Female","Female","Male","Male","Male","Female","Male","Male","Male","Female","Female","Female","Female","Male","Female","Female","Female","Female","Female","Female","Female","Female","Female","Male","Male","Male","Female","Female","Female","Male","Female","Male","Male","Female","Female","Female","Male","Male","Female","Male","Female","Male","Female","Male","Male","Female","Male","Female","Male","Male","Female","Male","Male","Female","Female","Male","Male","Male","Female","Female","Male","Male","Female","Male","Male","Male","Female","Female","Male","Female","Female","Female","Female","Female","Female","Male","Male","Female","Male","Male","Male","Male","Female","Male","Female","Female","Male","Male","Female","Male","Female","Female","Female","Male","Female","Male","Female","Female","Male","Male","Female","Male","Female","Male","Male","Female","Male","Female","Male","Male","Female","Male","Female","Female","Female","Male","Male","Male","Male","Male","Male","Female","Male","Female","Male","Male","Male","Male","Male","Male","Male","Male","Male","Female","Female","Male","Female","Female","Female","Male","Male","Male","Male","Female","Female","Female","Female","Female","Male","Female","Male","Male","Male","Female","Female","Male","Male","Male","Female","Male","Male","Male","Male","Male","Female","Male","Male","Female","Male","Male","Female","Male","Male","Male","Male","Male","Male","Male","Male","Female","Female","Female","Male","Male","Female","Female","Female","Female","Male","Female","Male","Female","Male","Female","Male","Female","Male","Male","Male","Male","Male","Male","Male","Female","Female","Female","Male","Female","Female","Female","Male","Male","Female","Male","Female","Female","Male","Male","Male","Female","Female","Male","Female","Female","Male","Male","Male","Male","Male","Female","Female","Male","Female","Male","Female","Female","Male","Female","Female","Male","Female","Male","Male","Female","Female","Female","Female","Male","Female","Female","Male","Female","Female","Male","Female","Female","Female","Male","Female","Male","Male","Female","Male","Male","Female","Female","Male","Female","Female","Male","Female","Female","Male","Male","Female","Female","Female","Male","Male","Female","Female","Male","Male","Female","Female","Female","Male","Female","Female","Male","Female","Male","Male","Male","Male","Male","Male","Male","Male","Female","Male","Female","Female","Female","Female","Female","Female","Female","Female","Female","Female","Female","Female","Male","Female","Female","Male","Female","Female","Male","Male","Male","Male","Male","Female","Male","Male","Male","Male","Female","Male","Female","Male","Female","Female","Male","Female","Male","Female","Male","Female","Male","Male","Female","Male","Female","Female","Female","Female","Female","Female","Male","Male","Female","Male","Female","Male","Male","Male","Female","Female","Male","Male","Female","Male","Male","Male","Male","Male","Male","Female","Female","Male","Male","Female","Male","Male","Female","Female","Female","Male","Female","Female","Female","Male","Male","Female","Male","Male","Male","Female","Male","Male","Male","Male","Female","Female","Male","Male","Male","Female","Male","Male","Male","Male","Female","Female","Male","Male","Female","Male","Male","Female","Female","Female","Male","Male","Male","Male","Female","Male","Female","Male","Male","Male","Male","Male","Male","Female","Male","Female","Female","Male","Male","Female","Female","Male","Male","Female","Male","Female","Female","Male","Male","Male","Female","Male","Female","Female","Female","Female","Female","Female","Male","Male","Female","Male","Female","Male","Female","Male","Female","Female","Male","Female","Male","Male","Female","Male","Male","Male","Male","Female","Male","Female","Male","Male","Female","Female","Female","Female","Female","Male","Female","Male","Male","Female","Female","Male","Male","Female","Male","Male","Male","Male","Male","Female","Female","Female","Female","Male","Female","Male","Female","Male","Female","Female","Female","Male","Female","Female","Female","Female","Female","Male","Female","Male","Male","Female","Female","Male","Female","Male","Female","Male","Female","Male","Male","Female","Male","Male","Male","Female","Male","Female","Male","Male","Female","Female","Male","Male","Female","Male","Female","Female","Female","Male","Female","Female","Male","Female","Female","Female","Male","Male","Male","Male","Female","Female","Female","Female","Female","Female","Male","Male","Female","Female","Male","Male","Female","Male","Male","Male","Female","Female","Female","Female","Female","Female","Male","Female","Female","Male","Male","Female","Male","Female","Male","Female","Female","Male","Male","Male","Male","Female","Male","Male","Female","Male","Male","Female","Male","Female","Male","Female","Female","Female","Female","Male","Male","Female","Male","Female","Female","Male","Male","Female","Male","Male","Male","Male","Female","Male","Female","Female","Female","Female","Female","Female","Female","Female","Female","Male","Male","Female","Female","Male","Female","Male","Male","Female","Male","Female","Female","Female","Male","Male","Male","Male","Female","Female","Female","Male","Male","Male","Male","Male","Female","Female","Female","Female","Female","Female","Male","Female","Female","Male","Female","Male","Female","Female","Female","Female","Male","Female","Female","Female","Female","Male","Male","Female","Female","Male","Male","Female","Male","Male","Male","Male","Male","Female","Male","Female","Female","Female","Male","Female","Female","Female","Female","Female","Male","Female","Female","Female","Female","Female","Male","Male","Female","Male","Male","Female","Male","Male","Male","Male","Male","Female","Female","Female","Male","Male","Male","Female","Female","Male","Female","Male","Female","Male","Female","Male","Male","Male","Male","Male","Female","Male","Male","Male","Male","Male","Female","Female","Female","Male","Female","Female","Male","Female","Female","Male","Female","Female","Female","Male","Male","Female","Male","Male","Female","Male","Male","Female","Male","Male","Male","Female","Male","Male","Male","Female","Male","Male","Female","Male","Male","Male","Male","Female","Male","Female","Male","Male","Female","Female","Female","Male","Male","Male","Male","Male","Female","Female","Female","Male","Male","Female","Male","Female","Female","Female","Female","Female","Male","Male","Female","Female","Female","Female","Female","Male","Female","Female","Male","Female","Male","Female","Female","Male","Male","Male","Female","Male","Female","Female","Male","Male","Male","Male","Male","Female","Male","Male","Male","Male","Female","Male","Female","Male","Male","Female","Male","Male","Female","Female","Female","Female","Female","Male","Male","Male","Male","Female","Female","Female","Male","Male","Male","Male","Female","Female","Male","Female","Female","Male","Male","Male","Female","Female","Female","Female","Male","Male","Male","Female","Female","Male","Male","Female","Male","Male","Female","Female","Male","Female","Male","Female","Female","Female","Female","Male","Male","Male","Female","Male","Female","Male","Male","Female","Male","Female","Female","Female","Female","Female","Female","Male","Female","Female","Male","Female","Female","Male","Male","Female","Female","Female","Male","Female","Male","Female","Male","Female","Male","Female","Male","Female","Male","Female","Male","Male","Male","Female","Female","Female","Male","Male","Male","Female","Female","Female","Male","Male","Female","Male","Female","Female","Female","Female","Male","Male","Male","Male","Female","Female","Female","Male","Female","Female","Female","Male","Male","Female","Male","Female","Male","Female","Female","Female","Male","Female","Male","Male","Female","Female","Female","Male","Male","Male","Female","Male","Female","Male","Male","Female","Male","Male","Female","Female","Female","Female","Female","Female","Female","Male","Female","Male","Male","Male","Female","Male","Male","Female","Male","Female","Male","Female","Male","Female","Male","Female","Female","Female","Male","Female","Male","Male","Female","Male","Female","Male","Female","Female","Male","Female","Female","Male","Female","Male","Male","Female","Male","Male","Male","Male","Male","Male","Female","Female","Male","Female","Female","Female","Male","Female","Female","Male","Female","Male","Male","Male","Male","Male","Male","Female","Female","Female","Male","Male","Female","Female","Male","Male","Female","Male","Male","Female","Female","Male","Male","Female","Male","Male","Female","Male","Male","Male","Male","Female","Female","Male","Male","Male","Male","Female","Male","Female","Female","Male","Female","Female","Female","Male","Female","Male","Male","Male","Female","Male","Female","Female","Female","Male","Male","Female","Male","Female","Female","Male","Female","Female","Male","Female","Male","Female","Male","Female","Male","Female","Female","Female","Female","Male","Female","Female","Female","Female","Male","Female","Male","Female","Male","Female","Male","Male","Male","Male","Male","Female","Female","Male","Female","Male","Female","Female","Male","Female","Male","Male","Female","Male","Male","Female","Female","Male","Male","Female","Female","Male","Female","Female","Female","Male","Female","Male","Male","Male","Male","Male","Female","Male","Male","Female","Female","Female","Male","Female","Male","Male","Male","Female","Male","Female","Female","Male","Female","Male","Male","Female","Female","Male","Female","Male","Male","Female","Male","Female","Female","Female","Female","Female","Female","Female","Female","Female","Female","Female","Male","Female","Male","Female","Male","Male","Female","Male","Male","Male","Female","Male","Male","Male","Male","Female","Female","Female","Male","Female","Male","Female","Male","Male","Male","Female","Male","Male","Male","Female","Female","Male","Female","Female","Male","Male","Female","Female","Female","Female","Male","Male","Female","Male","Male","Female","Male","Male","Male","Female","Female","Male","Male","Male","Female","Female","Female","Female","Female","Female","Female","Male","Female","Female","Female","Female","Male","Female","Female","Female","Female","Female","Female","Male","Female","Female","Male","Female","Male","Male","Male","Male","Female","Female","Male","Male","Female","Male","Male","Female","Female","Male","Female","Female","Male","Female","Female","Male","Male","Female","Female","Male","Female","Female","Female","Male","Male","Male","Male","Male","Female","Male","Female","Female","Female","Male","Male","Male","Female","Female","Female","Male","Female","Male","Male","Male","Female","Male","Female","Male","Female","Male","Male","Female","Male","Male","Female","Male","Male","Male","Female","Female","Male","Female","Male","Female","Male","Male","Male","Female","Male","Male","Male","Male","Male","Male","Male","Female","Male","Male","Male","Male","Female","Male","Male","Male","Male","Female","Male","Male","Female","Male","Male","Male","Female","Female","Female","Male","Female","Female","Female","Female","Male","Female","Male","Male","Male","Female","Female","Male","Male","Female","Male","Female","Female","Male","Female","Male","Male","Female","Female","Male","Female","Female","Male","Female","Female","Male","Female","Male","Male","Female","Female","Female","Female","Female","Female","Female","Female","Male","Female","Female","Male","Male","Female","Male","Male","Male","Male","Male","Female","Female","Female","Female","Male","Female","Male","Female","Male","Male","Female","Female","Female","Male","Female","Female","Male","Male","Female","Male","Male","Female","Female","Male","Female","Female","Male","Female","Male","Male","Female","Female","Male","Male","Male","Female","Female","Male","Male","Female","Female","Male","Female","Male","Male","Male","Female","Female","Female","Female","Male","Female","Male","Female","Male","Male","Female","Male","Female","Male","Female","Female","Male","Male","Male","Male","Male","Male","Male","Female","Male","Male","Female","Female","Male","Female","Female","Female","Female","Female","Female","Female","Female","Male","Female","Male","Male","Male","Female","Female","Female","Female","Male","Male","Female","Female","Female","Female","Female","Male","Male","Male","Female","Male","Female","Male","Male","Female","Female","Male","Female","Female","Male","Male","Male","Male","Male","Female","Male","Male","Male","Male","Female","Male","Female","Male","Male","Female","Male","Male","Female","Male","Female","Female","Male","Male","Female","Male","Female","Female","Male","Female","Female","Female","Male","Female","Female","Male","Male","Male","Male","Male","Female","Female","Female","Male","Female","Female","Female","Female","Female","Male","Female","Female","Female","Female","Male","Male","Female","Female","Female","Male","Male","Female","Male","Male","Female","Male","Female","Female","Female","Male","Female","Male","Female","Female","Female","Male","Male","Female","Female","Female","Female","Male","Female","Female","Male","Female","Male","Male","Male","Male","Female","Female","Male","Male","Female","Female","Female","Male","Female","Female","Female","Female","Male","Male","Male","Male","Male","Female","Female","Male","Female","Female","Male","Male","Male","Female","Female","Female","Female","Male","Female","Female","Female","Female","Female","Male","Male","Male","Male","Female","Female","Female","Male","Male","Male","Female","Female","Female","Female","Female","Female","Male","Male","Male","Male","Female","Female","Male","Male","Female","Female","Male","Male","Male","Female","Male","Female","Female","Male","Male","Male","Female","Female","Male","Female","Female","Female","Female","Female","Female","Female","Male","Female","Male","Female","Female","Female","Male","Male","Female","Female","Female","Male","Male","Female","Female","Female","Female","Male","Male","Female","Female","Male","Female","Male","Male","Male","Female","Male","Male","Male","Male","Male","Male","Female","Female","Female","Male","Male","Male","Male","Female","Female","Female","Female","Male","Female","Female","Male","Female","Male","Male","Female","Male","Male","Male","Female","Male","Male","Female","Male","Male","Female","Female","Male","Male","Female","Female","Female","Male","Female","Male","Female","Male","Male","Female","Female","Female","Female","Female","Male","Male","Male","Female","Male","Female","Male","Male","Female","Female","Female","Female","Female","Male","Male","Male","Male","Female","Male","Female","Female","Male","Male","Female","Male","Female","Female","Female","Female","Female","Male","Male","Female","Female","Female","Male","Female","Male","Male","Male","Male","Female","Male","Male","Male","Female","Female","Male","Male","Female","Female","Male","Female","Male","Male","Male","Female","Female","Female","Female","Female","Male","Male","Female","Male","Male","Male","Female","Male","Male","Male","Male","Female","Male","Female","Male","Male","Female","Male","Male","Male","Male","Male","Male","Female","Female","Female","Male","Male","Female","Female","Male","Female","Female","Female","Female","Female","Female","Male","Male","Female","Female","Female","Female","Male","Male","Female","Female","Female","Female","Female","Female","Female","Male","Female","Male","Male","Female","Male","Male","Female","Female","Male","Female","Male","Male","Female","Male","Male","Female","Female","Female","Male","Female","Female","Female","Male","Male","Male","Female","Female","Female","Male","Male","Female","Female","Female","Female","Male","Male","Female","Male","Female","Female","Female","Female","Male","Male","Male","Male","Male","Male","Male","Female","Male","Male","Female","Male","Female","Male","Male","Female","Male","Female","Male","Male","Male","Male","Female","Female","Female","Female","Male","Female","Female","Male","Female","Female","Male","Male","Female","Female","Male","Male","Male","Male","Female","Male","Male","Male","Male","Female","Male","Female","Male","Female","Female","Female","Female","Male","Male","Female","Female","Male","Female","Female","Female","Male","Male","Female","Female","Female","Female","Female","Male","Female","Male","Female","Female","Male","Male","Female","Male","Female","Male","Male","Male","Female","Male","Female","Female","Female","Male","Female","Female","Male","Male","Female","Female","Female","Male","Male","Female","Female","Female","Male","Male","Male","Male","Female","Female","Male","Male","Female","Female","Male","Female","Female","Female","Female","Male","Female","Male","Female","Male","Female","Male","Female","Male","Female","Female","Male","Female","Female","Male","Male","Female","Male","Female","Female","Male","Female","Female","Male","Male","Female","Male","Male","Male","Male","Male","Female","Male","Male","Female","Female","Female","Male","Male","Male","Female","Male","Male","Male","Male","Male","Male","Female","Male","Female","Male","Female","Male","Male","Female","Female","Female","Female","Male","Male","Male","Male","Female","Male","Female","Female","Male","Female","Male","Female","Male","Male","Female","Male","Female","Female","Male","Female","Male","Male","Female","Male","Female","Female","Female","Female","Male","Male","Male","Female","Male","Female","Male","Male","Male","Male","Female","Female","Male","Female","Female","Female","Male","Male","Female","Female","Male","Female","Male","Female","Female","Female","Female","Male","Male","Female","Female","Male","Female","Male","Female","Female","Male","Male","Female","Female","Male","Male","Female","Female","Male","Female","Male","Male","Male","Male","Female","Female","Female","Male","Female","Male","Male","Female","Male","Female","Male","Male","Male","Female","Female","Male","Female","Female","Female","Female","Male","Male","Female","Male","Male","Female","Male","Female","Male","Male","Male","Female","Female","Male","Female","Male","Female","Male","Male","Male","Female","Female","Male","Male","Male","Male","Male","Male","Female","Male","Male","Male","Male","Female","Female","Female","Male","Female","Female","Female","Male","Male","Male","Male","Male","Male","Female","Male","Female","Female","Female","Male","Male","Female","Female","Male","Female","Female","Female","Male","Female","Male","Male","Male","Female","Male","Female","Female","Female","Female","Female","Male","Male","Male","Male","Male","Male","Male","Male","Male","Female","Female","Male","Female","Male","Male","Male","Female","Male","Male","Female","Male","Female","Female","Female","Female","Female","Female","Female","Male","Female","Male","Male","Male","Male","Female","Female","Female","Female","Male","Male","Female","Female","Female","Female","Female","Male","Female","Male","Female","Male","Female","Male","Male","Male","Male","Male","Male","Male","Female","Male","Female","Female","Male","Female","Female","Female","Female","Male","Male","Female","Male","Female","Female","Male","Female","Male","Female","Male","Female","Male","Female","Female","Female","Male","Female","Female","Female","Male","Female","Male","Male","Male","Male","Female","Female","Male","Female","Female","Female","Female","Male","Female","Female","Female","Female","Female","Male","Female","Male","Male","Male","Female","Male","Female","Female","Female","Male","Female","Male","Female","Male","Female","Male","Male","Female","Female","Male","Male","Male","Male","Male","Female","Male","Male","Female","Male","Female","Female","Female","Male","Male","Female","Male","Female","Male","Male","Female","Male","Male","Female","Female","Male","Female","Male","Male","Female","Male","Female","Male","Female","Female","Female","Female","Male","Male","Male","Female","Male","Male","Female","Female","Female","Female","Female","Female","Female","Male","Female","Female","Female","Female","Male","Male","Female","Female","Male","Male","Female","Female","Female","Female","Female","Female","Male","Male","Male","Female","Female","Female","Male","Male","Female","Male","Female","Male","Female","Male","Male","Male","Male","Female","Male","Male","Female","Male","Male","Male","Female","Male","Female","Male","Female","Male","Male","Female","Male","Male","Female","Female","Male","Male","Female","Male","Male","Male","Female","Female","Male","Female","Male","Male","Male","Male","Female","Male","Female","Male","Male","Male","Female","Male","Female","Female","Male","Male","Male","Female","Female","Male","Male","Female","Female","Female","Male","Female","Male","Female","Female","Male","Male","Male","Female","Male","Male","Female","Female","Female","Male","Male","Female","Male","Male","Male","Female","Female","Male","Male","Female","Male","Male","Male","Male","Male","Female","Male","Female","Female","Male","Female","Male","Female","Male","Female","Male","Male","Male","Male","Male","Male","Female","Male","Male","Female","Male","Male","Male","Female","Male","Female","Female","Male","Male","Male","Male","Male","Male","Female","Female","Female","Female","Female","Male","Male","Male","Female","Female","Female","Female","Female","Female","Female","Female","Male","Male","Male","Female","Female","Male","Male","Female","Male","Female","Male","Female","Male","Female","Female","Female","Male","Female","Female","Female","Male","Male","Female","Male","Male","Female","Female","Female","Female","Female","Male","Male","Female","Female","Female","Female","Female","Male","Male","Male","Male","Female","Male","Male","Male","Female","Female","Male","Female","Female","Female","Female","Female","Male","Male","Female","Male","Male","Female","Male","Male","Female","Female","Male","Male","Female","Female","Male","Male","Female","Male","Male","Female","Female","Female","Female","Male","Female","Female","Male","Male","Male","Female","Female","Female","Female","Male","Male","Female","Female","Male","Male","Female","Male","Female","Female","Male","Female","Male","Male","Female","Female","Male","Female","Female","Female","Male","Male","Male","Female","Female","Male","Male","Female","Male","Female","Male","Male","Female","Female","Female","Female","Female","Male","Female","Female","Male","Female","Male","Female","Male","Male","Female","Female","Male","Female","Female","Male","Male","Female","Male","Male","Female","Female","Female","Male","Male","Male","Male","Male","Male","Female","Female","Female","Male","Male","Male","Male","Male","Male","Male","Male","Male","Male","Female","Female","Female","Female","Male","Male","Male","Male","Male","Female","Female","Male","Female","Male","Male","Male","Male","Male","Male","Female","Female","Male","Female","Male","Female","Male","Male","Female","Female","Male","Male","Male","Female","Male","Male","Male","Male","Male","Male","Female","Female","Female","Male","Male","Male","Male","Male","Male","Male","Male","Male","Female","Female","Male","Female","Male","Male","Female","Male","Female","Female","Male","Female","Female","Male","Female","Female","Female","Male","Female","Male","Female","Female","Male","Female","Male","Female","Female","Female","Female","Male","Male","Female","Male","Male","Female","Male","Female","Female","Female","Male","Male","Female","Female","Male","Female","Male","Male","Male","Female","Female","Male","Male","Male","Female","Female","Female","Male","Female","Female","Female","Female","Female","Female","Female","Male","Female","Female","Female","Female","Female","Male","Male","Female","Female","Male","Male","Male","Male","Female","Female","Female","Female","Female","Female","Female","Male","Female","Female","Female","Male","Male","Female","Female","Female","Male","Female","Male","Male","Male","Female","Male","Male","Female","Female","Male","Male","Female","Female","Male","Male","Male","Male","Female","Female","Male","Male","Female","Male","Female","Female","Female","Female","Male","Male","Female","Male","Male","Male","Female","Female","Male","Male","Female","Female","Male","Male","Male","Male","Male","Male","Male","Male","Female","Male","Male","Female","Female","Female","Male","Male","Female","Male","Female","Male","Female","Male","Male","Female","Female","Female","Female","Male","Male","Female","Female","Female","Male","Male","Female","Male","Female","Female","Male","Female","Male","Female","Male","Male","Male","Female","Male","Female","Female","Male","Female","Male","Female","Male","Female","Male","Male","Female","Male","Male","Female","Female","Male","Female","Female","Female","Female","Female","Male","Male","Male","Male","Male","Male","Female","Male","Male","Male","Female","Male","Female","Male","Male","Male","Male","Male","Male","Male","Male","Female","Male","Male","Male","Male","Male","Male","Female","Female","Female","Female","Male","Male","Female","Female","Female","Female","Male","Male","Male","Male","Female","Female","Female","Male","Male","Female","Male","Male","Female","Male","Male","Male","Female","Male","Female","Female","Female","Male","Male","Male","Female","Female","Female","Male","Female","Female","Male","Male","Male","Female","Male","Female","Male","Male","Male","Female","Female","Female","Male","Male","Male","Male","Male","Male","Male","Female","Female","Male","Female","Male","Male","Male","Male","Female","Female","Female","Male","Female","Female","Female","Male","Male","Male","Female","Male","Female","Female","Female","Female","Male","Female","Female","Male","Female","Male","Male","Female","Male","Male","Male","Male","Male","Male","Female","Female","Female","Male","Female","Female","Female","Female","Male","Female","Male","Female","Female","Female","Female","Female","Female","Female","Male","Female","Female","Female","Male","Male","Male","Male","Male","Male","Female","Female","Female","Female","Male","Female","Male","Male","Male","Female","Male","Male","Male","Male","Male","Male","Female","Male","Male","Male","Female","Male","Female","Male","Female","Male","Male","Male","Female","Male","Female","Female","Female","Male","Male","Female","Female","Female","Male","Female","Male","Male","Male","Male","Female","Male","Female","Male","Female","Male","Female","Female","Male","Female","Female","Male","Female","Female","Female","Male","Female","Male","Male","Male","Female","Female","Female","Female","Male","Female","Female","Female","Female","Female","Female","Male","Female","Male","Female","Female","Female","Male","Female","Female","Male","Male","Male","Female","Female","Male","Female","Female","Female","Male","Female","Female","Female","Male","Female","Male","Female","Male","Female","Male","Male","Male","Female","Female","Male","Female","Female","Female","Male","Female","Female","Male","Male","Male","Female","Female","Male","Male","Male","Male","Female","Female","Male","Female","Female","Female","Female","Female","Male","Male","Female","Female","Male","Male","Female","Male","Female","Male","Male","Female","Female","Male","Male","Male","Female","Female","Female","Female","Male","Male","Male","Female","Female","Male","Male","Male","Male","Male","Male","Male","Male","Male","Male","Female","Female","Male","Male","Female","Male","Female","Male","Male","Male","Female","Male","Female","Male","Female","Male","Male","Male","Female","Female","Female","Male","Female","Male","Male","Female","Female","Male","Male","Male","Male","Female","Male","Male","Female","Female","Female","Female","Female","Male","Male","Female","Female","Male","Male","Male","Male","Male","Female","Female","Female","Male","Male","Male","Female","Male","Female","Female","Male","Female","Female","Male","Male","Male","Female","Male","Male","Male","Female","Female","Female","Male","Male","Male","Male","Female","Male","Male","Male","Female","Male","Male","Female","Female","Female","Male","Male","Male","Male","Female","Male","Female","Female","Male","Female","Female","Female","Female","Female","Female","Female","Female","Male","Male","Male","Female","Male","Female","Female","Female","Female","Female","Female","Female","Female","Female","Female","Female","Female","Female","Male","Male","Female","Female","Female","Male","Male","Female","Female","Female","Male","Male","Female","Male","Male","Male","Male","Female","Female","Male","Female","Male","Female","Female","Male","Female","Male","Female","Male","Female","Male","Female","Male","Female","Female","Male","Male","Male","Male","Female","Male","Male","Female","Male","Male","Female","Female","Male","Male","Male","Male","Female","Male","Female","Female","Male","Female","Female","Male","Female","Male","Male","Female","Male","Female","Male","Female","Male","Female","Female","Male","Female","Male","Male","Female","Female","Female","Female","Female","Male","Female","Female","Male","Male","Female","Male","Male","Female","Female","Female","Female","Female","Female","Male","Female","Male","Male","Female","Female","Female","Male","Male","Female","Female","Female","Female","Male","Female","Male","Female","Female","Male","Female","Female","Female","Male","Female","Male","Female","Male","Male","Male","Male","Male","Female","Male","Female","Female","Female","Female","Male","Male","Female","Female","Male","Female","Male","Male","Male","Female","Male","Female","Female","Male","Female","Female","Female","Male","Male","Male","Female","Male","Female","Female","Male","Female","Female","Male","Male"],"legendgroup":"","name":"","showlegend":true,"type":"pie","pull":[0.1],"rotation":90,"textinfo":"percent+label"}],                        {"template":{"data":{"barpolar":[{"marker":{"line":{"color":"white","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"rgb(36,36,36)"},"error_y":{"color":"rgb(36,36,36)"},"marker":{"line":{"color":"white","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"rgb(36,36,36)","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"rgb(36,36,36)"},"baxis":{"endlinecolor":"rgb(36,36,36)","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"rgb(36,36,36)"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"contour"}],"heatmapgl":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"heatmapgl"}],"heatmap":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"histogram2d"}],"histogram":[{"marker":{"line":{"color":"white","width":0.6}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scattermapbox"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scatterpolar"}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"rgb(237,237,237)"},"line":{"color":"white"}},"header":{"fill":{"color":"rgb(217,217,217)"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"colorscale":{"diverging":[[0.0,"rgb(103,0,31)"],[0.1,"rgb(178,24,43)"],[0.2,"rgb(214,96,77)"],[0.3,"rgb(244,165,130)"],[0.4,"rgb(253,219,199)"],[0.5,"rgb(247,247,247)"],[0.6,"rgb(209,229,240)"],[0.7,"rgb(146,197,222)"],[0.8,"rgb(67,147,195)"],[0.9,"rgb(33,102,172)"],[1.0,"rgb(5,48,97)"]],"sequential":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"sequentialminus":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]]},"colorway":["#1F77B4","#FF7F0E","#2CA02C","#D62728","#9467BD","#8C564B","#E377C2","#7F7F7F","#BCBD22","#17BECF"],"font":{"color":"rgb(36,36,36)"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"white","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"white","polar":{"angularaxis":{"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside"},"bgcolor":"white","radialaxis":{"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside"}},"scene":{"xaxis":{"backgroundcolor":"white","gridcolor":"rgb(232,232,232)","gridwidth":2,"linecolor":"rgb(36,36,36)","showbackground":true,"showgrid":false,"showline":true,"ticks":"outside","zeroline":false,"zerolinecolor":"rgb(36,36,36)"},"yaxis":{"backgroundcolor":"white","gridcolor":"rgb(232,232,232)","gridwidth":2,"linecolor":"rgb(36,36,36)","showbackground":true,"showgrid":false,"showline":true,"ticks":"outside","zeroline":false,"zerolinecolor":"rgb(36,36,36)"},"zaxis":{"backgroundcolor":"white","gridcolor":"rgb(232,232,232)","gridwidth":2,"linecolor":"rgb(36,36,36)","showbackground":true,"showgrid":false,"showline":true,"ticks":"outside","zeroline":false,"zerolinecolor":"rgb(36,36,36)"}},"shapedefaults":{"fillcolor":"black","line":{"width":0},"opacity":0.3},"ternary":{"aaxis":{"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside"},"baxis":{"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside"},"bgcolor":"white","caxis":{"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside"}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside","title":{"standoff":15},"zeroline":false,"zerolinecolor":"rgb(36,36,36)"},"yaxis":{"automargin":true,"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside","title":{"standoff":15},"zeroline":false,"zerolinecolor":"rgb(36,36,36)"}}},"legend":{"tracegroupgap":0},"title":{"text":"Gender"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('2dc69def-7fce-41b9-a10d-f42a5f35a8e6');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
fig = px.pie(df, names = 'seniorcitizen', template = 'simple_white', title = 'Senior Citizen')
fig.update_traces(rotation = 90, pull = [0.1], textinfo = 'percent+label')
fig.show()
```


<div>                            <div id="b2149a9a-8a49-41cf-b99a-001ada357311" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("b2149a9a-8a49-41cf-b99a-001ada357311")) {                    Plotly.newPlot(                        "b2149a9a-8a49-41cf-b99a-001ada357311",                        [{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"hovertemplate":"seniorcitizen=%{label}\u003cextra\u003e\u003c\u002fextra\u003e","labels":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,1,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,1,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,1,0,1,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,1,0,1,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,1,1,0,0,0,1,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,1,0,1,1,1,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,1,0,0,0,0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,1,1,0,0,0,1,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,1,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,0,0,1,0,0,1,1,0,0,0,0,0,0,1,0,1,1,0,0,0,1,1,1,0,1,0,0,0,0,1,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,1,1,0,0,0,1,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,1,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,1,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,0,1,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,1,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,0,1,0,1,0,1,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,1,0,0,0,0,1,0,0,1,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,1,1,0,0,0,0,1,1,0,0,0,0,1,0,1,0,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1,0,1,1,1,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,0,1,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,1,0,0,0,0,0,1,0,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,1,0,0,1,0,1,1,0,0,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,1,0],"legendgroup":"","name":"","showlegend":true,"type":"pie","pull":[0.1],"rotation":90,"textinfo":"percent+label"}],                        {"template":{"data":{"barpolar":[{"marker":{"line":{"color":"white","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"rgb(36,36,36)"},"error_y":{"color":"rgb(36,36,36)"},"marker":{"line":{"color":"white","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"rgb(36,36,36)","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"rgb(36,36,36)"},"baxis":{"endlinecolor":"rgb(36,36,36)","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"rgb(36,36,36)"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"contour"}],"heatmapgl":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"heatmapgl"}],"heatmap":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"histogram2d"}],"histogram":[{"marker":{"line":{"color":"white","width":0.6}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scattermapbox"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scatterpolar"}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"rgb(237,237,237)"},"line":{"color":"white"}},"header":{"fill":{"color":"rgb(217,217,217)"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"colorscale":{"diverging":[[0.0,"rgb(103,0,31)"],[0.1,"rgb(178,24,43)"],[0.2,"rgb(214,96,77)"],[0.3,"rgb(244,165,130)"],[0.4,"rgb(253,219,199)"],[0.5,"rgb(247,247,247)"],[0.6,"rgb(209,229,240)"],[0.7,"rgb(146,197,222)"],[0.8,"rgb(67,147,195)"],[0.9,"rgb(33,102,172)"],[1.0,"rgb(5,48,97)"]],"sequential":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"sequentialminus":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]]},"colorway":["#1F77B4","#FF7F0E","#2CA02C","#D62728","#9467BD","#8C564B","#E377C2","#7F7F7F","#BCBD22","#17BECF"],"font":{"color":"rgb(36,36,36)"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"white","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"white","polar":{"angularaxis":{"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside"},"bgcolor":"white","radialaxis":{"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside"}},"scene":{"xaxis":{"backgroundcolor":"white","gridcolor":"rgb(232,232,232)","gridwidth":2,"linecolor":"rgb(36,36,36)","showbackground":true,"showgrid":false,"showline":true,"ticks":"outside","zeroline":false,"zerolinecolor":"rgb(36,36,36)"},"yaxis":{"backgroundcolor":"white","gridcolor":"rgb(232,232,232)","gridwidth":2,"linecolor":"rgb(36,36,36)","showbackground":true,"showgrid":false,"showline":true,"ticks":"outside","zeroline":false,"zerolinecolor":"rgb(36,36,36)"},"zaxis":{"backgroundcolor":"white","gridcolor":"rgb(232,232,232)","gridwidth":2,"linecolor":"rgb(36,36,36)","showbackground":true,"showgrid":false,"showline":true,"ticks":"outside","zeroline":false,"zerolinecolor":"rgb(36,36,36)"}},"shapedefaults":{"fillcolor":"black","line":{"width":0},"opacity":0.3},"ternary":{"aaxis":{"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside"},"baxis":{"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside"},"bgcolor":"white","caxis":{"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside"}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside","title":{"standoff":15},"zeroline":false,"zerolinecolor":"rgb(36,36,36)"},"yaxis":{"automargin":true,"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside","title":{"standoff":15},"zeroline":false,"zerolinecolor":"rgb(36,36,36)"}}},"legend":{"tracegroupgap":0},"title":{"text":"Senior Citizen"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('b2149a9a-8a49-41cf-b99a-001ada357311');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
fig = px.pie(df, names = 'dependents', template = 'simple_white', title = 'Dependents')
fig.update_traces(rotation = 90, pull = [0.1], textinfo = 'percent+label')
fig.show()
```


<div>                            <div id="f1af40a0-e7cc-4b74-bd34-b651d59410ba" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("f1af40a0-e7cc-4b74-bd34-b651d59410ba")) {                    Plotly.newPlot(                        "f1af40a0-e7cc-4b74-bd34-b651d59410ba",                        [{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"hovertemplate":"dependents=%{label}\u003cextra\u003e\u003c\u002fextra\u003e","labels":["No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","Yes","No","Yes","Yes","No","No","No","No","No","Yes","No","Yes","Yes","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","Yes","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","Yes","Yes","Yes","No","No","No","Yes","No","No","Yes","Yes","No","No","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","Yes","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","No","No","Yes","No","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","Yes","No","Yes","Yes","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","Yes","No","No","No","Yes","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","Yes","Yes","Yes","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","Yes","No","Yes","No","Yes","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","Yes","Yes","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","No","No","No","No","Yes","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","Yes","Yes","No","Yes","No","No","No","Yes","Yes","No","Yes","No","No","No","Yes","No","No","Yes","No","Yes","Yes","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","Yes","No","No","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","No","No","No","No","Yes","Yes","No","No","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","No","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","Yes","No","No","Yes","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","No","No","Yes","No","No","Yes","Yes","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","No","Yes","No","No","Yes","Yes","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","Yes","No","Yes","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","Yes","Yes","No","No","No","Yes","Yes","No","No","Yes","No","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","Yes","No","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","Yes","No","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","Yes","Yes","No","No","Yes","No","Yes","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","No","Yes","No","No","Yes","No","No","No","No","Yes","No","Yes","No","Yes","No","Yes","No","No","No","Yes","Yes","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","Yes","No","No","No","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","No","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","No","Yes","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","Yes","No","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","No","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","No","Yes","No","No","Yes","Yes","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","No","Yes","Yes","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","Yes","No","No","No","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","No","No","Yes","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","Yes","No","No","Yes","No","No","Yes","Yes","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","No","No","No","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","Yes","No","No","Yes","Yes","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","Yes","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","Yes","No","No","Yes","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","Yes","No","Yes","No","No","Yes","No","No","Yes","No","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","Yes","No","Yes","Yes","No","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","Yes","No","No","Yes","Yes","No","Yes","No","No","Yes","No","Yes","Yes","No","No","Yes","No","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","Yes","No","No","Yes","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","Yes","Yes","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","Yes","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","No","Yes","No","No","No","No","No","Yes","No","Yes","Yes","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","Yes","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","Yes","No","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","Yes","No","Yes","Yes","Yes","No","No","No","Yes","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","Yes","Yes","No","No","No","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","Yes","No","Yes","No","No","Yes","No","No","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","No","No","No","No","Yes","No","Yes","No","No","Yes","No","Yes","No","No","Yes","No","No","No","Yes","Yes","No","No","Yes","No","No","Yes","No","Yes","No","No","No","Yes","No","No","Yes","No","No","Yes","No","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","Yes","Yes","No","No","No","Yes","Yes","No","Yes","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","Yes","Yes","Yes","No","Yes","No","No","No","Yes","Yes","Yes","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","Yes","Yes","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","Yes","No","Yes","No","No","Yes","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","No","Yes","No","Yes","No","No","No","Yes","Yes","No","No","No","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","Yes","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","Yes","No","No","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","Yes","No","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","Yes","No","Yes","No","Yes","No","No","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","No","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","Yes","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","Yes","No","No","No","Yes","Yes","No","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","Yes","No","No","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","No","No","No","Yes","Yes","No","No","No","No","Yes","No","Yes","Yes","No","No","No","No","Yes","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","No","No","No","Yes","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No"],"legendgroup":"","name":"","showlegend":true,"type":"pie","pull":[0.1],"rotation":90,"textinfo":"percent+label"}],                        {"template":{"data":{"barpolar":[{"marker":{"line":{"color":"white","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"rgb(36,36,36)"},"error_y":{"color":"rgb(36,36,36)"},"marker":{"line":{"color":"white","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"rgb(36,36,36)","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"rgb(36,36,36)"},"baxis":{"endlinecolor":"rgb(36,36,36)","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"rgb(36,36,36)"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"contour"}],"heatmapgl":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"heatmapgl"}],"heatmap":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"histogram2d"}],"histogram":[{"marker":{"line":{"color":"white","width":0.6}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scattermapbox"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scatterpolar"}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"rgb(237,237,237)"},"line":{"color":"white"}},"header":{"fill":{"color":"rgb(217,217,217)"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"colorscale":{"diverging":[[0.0,"rgb(103,0,31)"],[0.1,"rgb(178,24,43)"],[0.2,"rgb(214,96,77)"],[0.3,"rgb(244,165,130)"],[0.4,"rgb(253,219,199)"],[0.5,"rgb(247,247,247)"],[0.6,"rgb(209,229,240)"],[0.7,"rgb(146,197,222)"],[0.8,"rgb(67,147,195)"],[0.9,"rgb(33,102,172)"],[1.0,"rgb(5,48,97)"]],"sequential":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"sequentialminus":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]]},"colorway":["#1F77B4","#FF7F0E","#2CA02C","#D62728","#9467BD","#8C564B","#E377C2","#7F7F7F","#BCBD22","#17BECF"],"font":{"color":"rgb(36,36,36)"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"white","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"white","polar":{"angularaxis":{"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside"},"bgcolor":"white","radialaxis":{"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside"}},"scene":{"xaxis":{"backgroundcolor":"white","gridcolor":"rgb(232,232,232)","gridwidth":2,"linecolor":"rgb(36,36,36)","showbackground":true,"showgrid":false,"showline":true,"ticks":"outside","zeroline":false,"zerolinecolor":"rgb(36,36,36)"},"yaxis":{"backgroundcolor":"white","gridcolor":"rgb(232,232,232)","gridwidth":2,"linecolor":"rgb(36,36,36)","showbackground":true,"showgrid":false,"showline":true,"ticks":"outside","zeroline":false,"zerolinecolor":"rgb(36,36,36)"},"zaxis":{"backgroundcolor":"white","gridcolor":"rgb(232,232,232)","gridwidth":2,"linecolor":"rgb(36,36,36)","showbackground":true,"showgrid":false,"showline":true,"ticks":"outside","zeroline":false,"zerolinecolor":"rgb(36,36,36)"}},"shapedefaults":{"fillcolor":"black","line":{"width":0},"opacity":0.3},"ternary":{"aaxis":{"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside"},"baxis":{"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside"},"bgcolor":"white","caxis":{"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside"}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside","title":{"standoff":15},"zeroline":false,"zerolinecolor":"rgb(36,36,36)"},"yaxis":{"automargin":true,"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside","title":{"standoff":15},"zeroline":false,"zerolinecolor":"rgb(36,36,36)"}}},"legend":{"tracegroupgap":0},"title":{"text":"Dependents"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('f1af40a0-e7cc-4b74-bd34-b651d59410ba');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
ax = sns.countplot(x='churn', hue= 'gender', data=df, palette='PuBu')
total = float(len(df))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2.,
            height + 0.3,
            f'{height:.0f} ({height / total * 100:.1f}%)',
            ha="center")
```


    
![png](output_12_0.png)
    



```python
ax = sns.countplot(x='churn', hue= 'seniorcitizen', data=df, palette='PuBu')
total = float(len(df))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2.,
            height + 0.3,
            f'{height:.0f} ({height / total * 100:.1f}%)',
            ha="center")
```


    
![png](output_13_0.png)
    



```python
ax = sns.countplot(x='churn', hue= 'dependents', data=df, palette='deep')
total = float(len(df))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2.,
            height + 0.3,
            f'{height:.0f} ({height / total * 100:.1f}%)',
            ha="center")
```


    
![png](output_14_0.png)
    



```python
sns.boxplot(x='churn', y='tenure', data=df)
plt.title('Churn Rate')
plt.xlabel('churn')
plt.ylabel('tenure')
plt.show()
```


    
![png](output_15_0.png)
    



```python
ax = sns.countplot(x='churn', hue= 'phoneservice', data=df, palette='deep')
total = float(len(df))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2.,
            height + 0.3,
            f'{height:.0f} ({height / total * 100:.1f}%)',
            ha="center")
```


    
![png](output_16_0.png)
    



```python
ax = sns.countplot(x='churn', hue= 'multiplelines', data=df, palette='deep')
total = float(len(df))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2.,
            height + 0.3,
            f'{height:.0f} ({height / total * 100:.1f}%)',
            ha="center")
```


    
![png](output_17_0.png)
    



```python
ax = sns.countplot(x='churn', hue= 'internetservice', data=df, palette='deep')
total = float(len(df))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2.,
            height + 0.3,
            f'{height:.0f} ({height / total * 100:.1f}%)',
            ha="center")
```


    
![png](output_18_0.png)
    



```python
ax = sns.countplot(x='churn', hue= 'contract', data=df, palette='mako')
ax.set_ylabel('No. of Customers')
total = float(len(df))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2.,
            height + 0.3,
            f'{height:.0f} ({height / total * 100:.1f}%)',
            ha="center")
```


    
![png](output_19_0.png)
    



```python
sns.boxplot(x='churn', y='monthlycharges', data=df)
plt.title('Churn Rate')
plt.xlabel('churn')
plt.ylabel('monthlycharges')
plt.show()
```


    
![png](output_20_0.png)
    



```python
print(df.groupby(['churn', 'contract']).monthlycharges.describe().round(1))
```

                           count  mean   std   min   25%   50%    75%    max
    churn contract                                                          
    No    Month-to-month  2220.0  61.5  27.9  18.8  38.5  65.0   84.9  116.5
          One year        1307.0  62.5  31.7  18.2  24.8  64.8   91.2  118.6
          Two year        1647.0  60.0  34.5  18.4  23.8  63.3   89.8  118.8
    Yes   Month-to-month  1655.0  73.0  24.1  18.8  55.2  79.0   90.9  117.4
          One year         166.0  85.1  25.6  19.3  70.1  95.0  104.7  118.4
          Two year          48.0  86.8  28.9  19.4  73.6  97.3  108.4  116.2
    


```python
### Total churn rate
# Count the number of customers who churned and who did not, and calculate the overall churn rate
churn_count = df['churn'].value_counts()
total_customers = churn_count.iloc[0] + churn_count.iloc[1]
churn_rate = (churn_count.iloc[1] / total_customers) * 100

# Print the churn rate as a formatted string with 3 decimal places
print('The overall churn rate in the company is: {:.3f}%'.format(churn_rate))
```

    The overall churn rate in the company is: 26.537%
    


```python
# Sort the pivot table by the 'Churn' column in descending order

contract_churn = pd.pivot_table(data=df, index='contract', values='churn', aggfunc=lambda x: x.map({'Yes':1, 'No':0}).mean())

contract_churn.sort_values('churn', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>churn</th>
    </tr>
    <tr>
      <th>contract</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Month-to-month</th>
      <td>0.427097</td>
    </tr>
    <tr>
      <th>One year</th>
      <td>0.112695</td>
    </tr>
    <tr>
      <th>Two year</th>
      <td>0.028319</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Changing churn numbers to absolutes 
df['churn_rate'] = df['churn'].apply(lambda x: 0 if x=='No' else 1)
```


```python
df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>seniorcitizen</th>
      <th>dependents</th>
      <th>tenure</th>
      <th>phoneservice</th>
      <th>multiplelines</th>
      <th>internetservice</th>
      <th>contract</th>
      <th>monthlycharges</th>
      <th>churn</th>
      <th>churn_rate</th>
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
      <td>0</td>
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
      <td>1</td>
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
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>8</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>Month-to-month</td>
      <td>99.65</td>
      <td>Yes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>22</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>Month-to-month</td>
      <td>89.10</td>
      <td>No</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>10</td>
      <td>No</td>
      <td>No</td>
      <td>DSL</td>
      <td>Month-to-month</td>
      <td>29.75</td>
      <td>No</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>28</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>Month-to-month</td>
      <td>104.80</td>
      <td>Yes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>62</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>One year</td>
      <td>56.15</td>
      <td>No</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
cat_cols = ["gender","seniorcitizen","dependents","phoneservice","multiplelines"
                    ,"internetservice","contract"]


num_cols = ["tenure","monthlycharges"]

target_col = 'churn'
# spliting categorical columns into Nominal and Binary columns

nominal_cols = ['gender','internetservice','contract']

binary_cols = ['seniorcitizen','dependents','phoneservice','multiplelines','internetservice','churn_rate']
```


```python
### Churn vs univariables as Demographics
#1 Gender
churn_by_gender = df.groupby('gender')['churn_rate'].mean()

# Display the resulting Series that shows the proportion of customers who churned for each gender
print(churn_by_gender)
```

    gender
    Female    0.269209
    Male      0.261603
    Name: churn_rate, dtype: float64
    


```python
#2 Senior Citizen
churn_by_senior = df.groupby('seniorcitizen')['churn_rate'].mean()

#Display the resulting Series that shows the proportion of customers who churned by senority status
print(churn_by_senior)
```

    seniorcitizen
    0    0.236062
    1    0.416813
    Name: churn_rate, dtype: float64
    


```python
#3 Dependents
churn_by_dependents = df.groupby('dependents')['churn_rate'].mean()

#Display the resulting Series that shows the proportion of customers who churned by depentent status
print(churn_by_dependents)
```

    dependents
    No     0.312791
    Yes    0.154502
    Name: churn_rate, dtype: float64
    


```python
### Correlations and linearity
category_corr = df[['churn_rate', 'seniorcitizen', 'tenure', 'monthlycharges']].corr(method='pearson')

# Print the churn_corr DataFrame
category_corr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>churn_rate</th>
      <th>seniorcitizen</th>
      <th>tenure</th>
      <th>monthlycharges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>churn_rate</th>
      <td>1.000000</td>
      <td>0.150889</td>
      <td>-0.352229</td>
      <td>0.193356</td>
    </tr>
    <tr>
      <th>seniorcitizen</th>
      <td>0.150889</td>
      <td>1.000000</td>
      <td>0.016567</td>
      <td>0.220173</td>
    </tr>
    <tr>
      <th>tenure</th>
      <td>-0.352229</td>
      <td>0.016567</td>
      <td>1.000000</td>
      <td>0.247900</td>
    </tr>
    <tr>
      <th>monthlycharges</th>
      <td>0.193356</td>
      <td>0.220173</td>
      <td>0.247900</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plotting the heatmap Correlation
plt.figure(figsize=(8, 6))
sns.heatmap(category_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()
```


    
![png](output_31_0.png)
    



```python
## Historigram representing mean and churning likelyness
mean_churn_rate = df['churn_rate'].mean()
plt.hist(df['churn_rate'], bins =10, edgecolor='k', alpha=0.7)
plt.axvline(mean_churn_rate, color='r', linestyle='dashed', linewidth=1)
plt.title('Histogram of Churn Rates')
plt.xlabel('Churn Rate')
plt.ylabel('Frequency')
plt.show()
```


    
![png](output_32_0.png)
    



```python
df1 = df.copy()
```


```python
df1.columns
```




    Index(['gender', 'seniorcitizen', 'dependents', 'tenure', 'phoneservice',
           'multiplelines', 'internetservice', 'contract', 'monthlycharges',
           'churn', 'churn_rate'],
          dtype='object')




```python
print (len(df1))
```

    7043
    


```python
#### Key features indicating churn
# Drop 'churn_rate' and 'churn' columns
churn_dropped = df1.drop(['churn_rate', 'churn'], axis=1)

# Perform One-Hot Encoding on categorical columns
ohe = OneHotEncoder()
ohe.fit(churn_dropped.select_dtypes(include=['object']))
categorical_columns = churn_dropped.select_dtypes(include=['object']).columns
X_ohe = pd.DataFrame(ohe.transform(churn_dropped.select_dtypes(include=['object'])).toarray(),
                     columns=ohe.get_feature_names_out(categorical_columns),
                     index=churn_dropped.index)

# Concatenate numerical and One-Hot Encoded features into X
X = pd.concat([churn_dropped.select_dtypes(include=['float64', 'int64']), X_ohe], axis=1)

# Set y as the target variable (churn_rate)
y = df1['churn_rate']

# Fit the classifier on the entire dataset
clf =RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Get feature importances
importances = clf.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print feature importances
print(feature_importance_df.head(10).to_string(index=False))
```

                        Feature  Importance
                 monthlycharges    0.433275
                         tenure    0.314646
        contract_Month-to-month    0.077057
    internetservice_Fiber optic    0.035047
            internetservice_DSL    0.031967
              contract_Two year    0.024731
                  seniorcitizen    0.018605
              contract_One year    0.011044
                  dependents_No    0.009046
                    gender_Male    0.008105
    


```python
# Plot feature importances
plt.figure(figsize=(11, 8))
ax = sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))

# Add labels (numerical values) on top of each bar
for p in ax.patches:
    ax.annotate(f'{p.get_width():.3f}', (p.get_width() + 0.005, p.get_y() + p.get_height()/2), ha='left', va='center')

plt.xlabel('Rate of Affecting Customer Churn')
plt.ylabel('Feature')
plt.title('Top 10 Key Indicators of Churn')
plt.show()
```


    
![png](output_37_0.png)
    



```python
#### Strategies to retain customers
## Assigning Categorical x values
c_num = X.assign(churn_rate=df['churn_rate'])
## Calculating Correlation between churn and other variables
corr_matrix = c_num.corr()
churn_corr = corr_matrix['churn_rate'].sort_values(ascending=False)

##Print variables that are strongly relted with churn
print(churn_corr.head(10))
print()
# Print recommendations based on feature importance
print('Recommendations for customer retention strategies:\n')

for feature, correlation in churn_corr.items():
    if correlation > 0.4 and feature == 'contract_Month-to-month':
        print(f"High Positive Correlation with Churn Rate ({correlation:.4f}):")
        print("Offer longer-term contracts to reduce churn.\n")
    elif correlation > 0.3 and feature == 'internetservice_Fiber optic':
        print(f"Moderate Positive Correlation with Churn Rate ({correlation:.4f}):")
        print("Evaluate service quality and consider offering alternative internet options.\n")
    elif correlation > 0.1 and feature == 'monthlycharges':
        print(f"Weak Positive Correlation with Churn Rate ({correlation:.4f}):")
        print("Review pricing strategies to optimize customer retention.\n")
    elif correlation < -0.1:
        if feature == 'dependents_No':
            print(f"Weak Negative Correlation with Churn Rate ({correlation:.4f}):")
            print("Develop family-oriented packages or incentives to retain customers with dependents.\n")
        elif feature == 'seniorcitizen':
            print(f"Moderate Negative Correlation with Churn Rate ({correlation:.4f}):")
            print("Tailor promotions and services to meet the needs of senior citizens.\n")
        elif feature == 'gender_Female' or feature == 'gender_Male':
            print(f"Weak Negative Correlation with Churn Rate ({correlation:.4f}):")
            print("Ensure services and promotions appeal equally to all gender demographics.\n")
    elif correlation > -0.1 and correlation < 0.1:
        if feature == 'multiplelines_Yes':
            print(f"Low Correlation with Churn Rate ({correlation:.4f}):")
            print("Enhance customer support and service for customers with multiple lines.\n")
        elif feature == 'phoneservice_Yes':
            print(f"Low Correlation with Churn Rate ({correlation:.4f}):")
            print("Ensure consistent and reliable phone service offerings.\n")
```

    churn_rate                     1.000000
    contract_Month-to-month        0.405103
    internetservice_Fiber optic    0.308020
    monthlycharges                 0.193356
    dependents_No                  0.164221
    seniorcitizen                  0.150889
    multiplelines_Yes              0.040102
    phoneservice_Yes               0.011942
    gender_Female                  0.008612
    gender_Male                   -0.008612
    Name: churn_rate, dtype: float64
    
    Recommendations for customer retention strategies:
    
    High Positive Correlation with Churn Rate (0.4051):
    Offer longer-term contracts to reduce churn.
    
    Moderate Positive Correlation with Churn Rate (0.3080):
    Evaluate service quality and consider offering alternative internet options.
    
    Weak Positive Correlation with Churn Rate (0.1934):
    Review pricing strategies to optimize customer retention.
    
    Low Correlation with Churn Rate (0.0401):
    Enhance customer support and service for customers with multiple lines.
    
    Low Correlation with Churn Rate (0.0119):
    Ensure consistent and reliable phone service offerings.
    
    


```python
df1.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>seniorcitizen</th>
      <th>tenure</th>
      <th>monthlycharges</th>
      <th>churn_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7043.000000</td>
      <td>7043.000000</td>
      <td>7043.000000</td>
      <td>7043.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.162147</td>
      <td>32.371149</td>
      <td>64.761692</td>
      <td>0.265370</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.368612</td>
      <td>24.559481</td>
      <td>30.090047</td>
      <td>0.441561</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>18.250000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>9.000000</td>
      <td>35.500000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>29.000000</td>
      <td>70.350000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>55.000000</td>
      <td>89.850000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>72.000000</td>
      <td>118.750000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Label Encoding to Numeric
lenc = LabelEncoder()
df2 = df_transformed.copy()
for i in df_transformed.columns:
    df2[i] = lenc.fit_transform(df2[i])
```


```python
df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>seniorcitizen</th>
      <th>dependents</th>
      <th>tenure</th>
      <th>phoneservice</th>
      <th>multiplelines</th>
      <th>monthlycharges</th>
      <th>churn</th>
      <th>churn_rate</th>
      <th>internetservice_DSL</th>
      <th>internetservice_Fiber optic</th>
      <th>contract_Month-to-month</th>
      <th>contract_One year</th>
      <th>contract_Two year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>142</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>498</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>436</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>0</td>
      <td>0</td>
      <td>266</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>729</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.drop(['churn_rate'],axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>seniorcitizen</th>
      <th>dependents</th>
      <th>tenure</th>
      <th>phoneservice</th>
      <th>multiplelines</th>
      <th>monthlycharges</th>
      <th>churn</th>
      <th>internetservice_DSL</th>
      <th>internetservice_Fiber optic</th>
      <th>contract_Month-to-month</th>
      <th>contract_One year</th>
      <th>contract_Two year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>142</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>498</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>436</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>0</td>
      <td>0</td>
      <td>266</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>729</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7038</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>991</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7039</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>72</td>
      <td>0</td>
      <td>0</td>
      <td>1340</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7040</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>137</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7041</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>795</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7042</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>66</td>
      <td>0</td>
      <td>0</td>
      <td>1388</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>7043 rows × 13 columns</p>
</div>




```python
from scipy.stats import skew, kurtosis
# Plot histograms and calculate skewness and kurtosis
numerical_columns = ['tenure', 'monthlycharges', 'churn']

# Calculate and print skewness and kurtosis
for column in numerical_columns:
    column_skewness = skew(df2[column])
    column_kurtosis = kurtosis(df2[column])
    print(f'{column} - Skewness: {column_skewness:.3f}, Kurtosis: {column_kurtosis:.3f}')

# Plot histograms with skewness and kurtosis in titles
plt.figure(figsize=(15, 5))
for i, column in enumerate(numerical_columns):
    plt.subplot(1, len(numerical_columns), i + 1)
    sns.histplot(df2[column], kde=True)
    plt.title(f'{column}\nSkewness: {skew(df2[column]):.3f} | Kurtosis: {kurtosis(df2[column]):.3f}')
plt.tight_layout()
plt.show()
```

    tenure - Skewness: 0.239, Kurtosis: -1.387
    monthlycharges - Skewness: 0.014, Kurtosis: -1.300
    churn - Skewness: 1.063, Kurtosis: -0.870
    


    
![png](output_43_1.png)
    



```python
df2.dtypes
```




    gender                         int64
    seniorcitizen                  int64
    dependents                     int64
    tenure                         int64
    phoneservice                   int64
    multiplelines                  int64
    monthlycharges                 int64
    churn                          int32
    churn_rate                     int64
    internetservice_DSL            int64
    internetservice_Fiber optic    int64
    contract_Month-to-month        int64
    contract_One year              int64
    contract_Two year              int64
    dtype: object




```python
dropped_churnrate = df2.drop(['churn_rate'], axis=1)
```


```python
dropped_churnrate.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>seniorcitizen</th>
      <th>dependents</th>
      <th>tenure</th>
      <th>phoneservice</th>
      <th>multiplelines</th>
      <th>monthlycharges</th>
      <th>churn</th>
      <th>internetservice_DSL</th>
      <th>internetservice_Fiber optic</th>
      <th>contract_Month-to-month</th>
      <th>contract_One year</th>
      <th>contract_Two year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>142</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>498</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>436</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>0</td>
      <td>0</td>
      <td>266</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>729</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_cal = dropped_churnrate.copy()
```


```python
columns_to_evaluate = ['tenure', 'monthlycharges']
churn_summary = df_cal.groupby('churn')[columns_to_evaluate].mean()
print(churn_summary) 
```

              tenure  monthlycharges
    churn                           
    0      37.569965      630.747971
    1      17.979133      829.628143
    


```python
sns.lmplot(x='tenure'
           ,y='monthlycharges'
           ,data=df_cal
           ,hue='churn'
            ,fit_reg=False
            ,markers=["o", "x"]
            ,palette= 'plasma')
plt.show()
```


    
![png](output_49_0.png)