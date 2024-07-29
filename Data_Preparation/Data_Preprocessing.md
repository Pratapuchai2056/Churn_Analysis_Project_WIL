
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
```df = pd.read_csv('Dataset(ATS).csv')
df.head(15)
```
**3. Displaying the Initial Data:**
To understand the shape of the dataset, we then will show the first fifteen rows and the
facts styles of every column.
```
df.dtypes
```
**4. Checking for the Missing Values and Standardizing the Column Names:**
If any nuls were present is to be removed using **drop()
```
df.isnull().sum()
df.columns
```
**5. Basic Visualizations:** **EDA
We will method our fifth step by using then creating the pie charts and through counting
the plots to visualize the distribution of categorical variables.
```
fig = px.pie(df, names='churn', template='simple_white', title='Churn')
fig.update_traces(rotation=90, pull=[0.1], textinfo='percent+label')
fig.show()
```
```
fig = px.pie(df, names='gender', template='simple_white', title='Gender')
fig.update_traces(rotation=90, pull=[0.1], textinfo='percent+label')
fig.show()
fig = px.pie(df, names='seniorcitizen', template='simple_white', title='Senior Citizen')
fig.update_traces(rotation=90, pull=[0.1], textinfo='percent+label')
fig.show()
ax = sns.countplot(x='churn', hue='gender', data=df, palette='PuBu')
total = float(len(df))
for p in ax.patches:
 height = p.get_height()
 ax.text(p.get_x() + p.get_width() / 2., height + 0.3, f'{height:.0f} ({height / total *
100:.1f}%)', ha='center')

plt.show()
ax = sns.countplot(x='churn', hue='contract', data=df, palette='mako')
ax.set_ylabel('No. of Customers')
total = float(len(df))
for p in ax.patches:
 height = p.get_height()
 ax.text(p.get_x() + p.get_width() / 2., height + 0.3, f'{height:.0f} ({height / total *
100:.1f}%)', ha='center')
plt.show()
6. Calculations:
On the sixth step we can show the descriptive data of the month-to-month prices columns
that is grouped by using churn and agreement.
print(df.groupby(['churn', 'contract']).monthlycharges.describe().round(1))
7. Data Preprocessing:
In the very last step, we can preprocess the information by manually mapping express
values and via appearing one-hot encoding.
df_transformed = df.copy()
columns1 = ['gender', 'dependents', 'churn', 'phoneservice', 'multiplelines']
for i in columns1:
 if i == 'gender':
 df_transformed[i] = df_transformed[i].map({'Female': 0, 'Male': 1})
 else:
 df_transformed[i] = df_transformed[i].map({'Yes': 1, 'No': 0})

columns2 = ['internetservice', 'contract']
df_transformed = pd.get_dummies(df_transformed, columns=columns2)
Conclusion:
In end, the above steps have efficiently confirmed how to load the dataset and displayed its
structure and then checked for lacking values and have helped us to standardize the column
names. These steps are very important for ensuring the records are easy and geared up to
apply for similarly analysis and modeling.