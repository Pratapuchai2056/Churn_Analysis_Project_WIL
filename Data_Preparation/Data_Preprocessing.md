
## Preparation and preprocessing of the dataset

# Introduction
The following part mentioned below would showcase us the process that is needed for the
preparing and preprocessing of the dataset according to our assignment.

**1. Importing the Libraries:**
We will first start our process by always importing the necessary libraries that are needed for data manipulation and preprocessing.
```
import pandas as pd
import sklearn as sl
from sklearn. preprocessing import LabelEncoder, StandardScaler
import numpy as np
```
 **2. Loading of the Dataset:**
Then the second step would be loading the dataset into a pandas Data Frame from a CSV file
named 'Dataset (ATS).csv'.
```
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

**3. Displaying the Initial Data:**
To understand the shape of the dataset, we then will show the first fifteen rows and the
facts styles of every column.
```
df.dtypes
```

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

```
```
df.count()
```
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
```
```
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

**4. Checking for the Missing Values and Standardizing the Column Names:**
If any nuls were present is to be removed using **drop()
```
df.isnull().sum()
print(df.columns)
```
```
    Index(['gender', 'SeniorCitizen', 'Dependents', 'tenure', 'PhoneService',
           'MultipleLines', 'InternetService', 'Contract', 'MonthlyCharges',
           'Churn'],
          dtype='object')
```

```
df_clean = df.dropna()
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

**5. Data Preprocessing:**
We will now perform encoding boolean values to binary values and perform one hot emcoding.
```
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_features = encoder.fit_transform(df[['gender', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'Contract']])
df_encoded = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())
df_encoded = pd.concat([df[['SeniorCitizen', 'tenure', 'MonthlyCharges']], df_encoded], axis=1)
df_encoded.head(50)
```
In end, the above steps have efficiently confirmed how to load the dataset and displayed its
structure and then checked for lacking values and have helped us to standardize the column
names. These steps are very important for ensuring the records are easy and geared up to
apply for similarly analysis and modeling.
