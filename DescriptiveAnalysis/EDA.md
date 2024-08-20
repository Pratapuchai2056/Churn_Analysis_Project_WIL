# Exploratory Data Analysis
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
```


```python
df = pd.read_csv('Dataset(ATS).csv')
```

**Data Exploration after inspection**
```python
df.head(15)
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



**Data Cleaning**
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
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.columns
```




    Index(['gender', 'seniorcitizen', 'dependents', 'tenure', 'phoneservice',
           'multiplelines', 'internetservice', 'contract', 'monthlycharges',
           'churn'],
          dtype='object')
# Viualization of both numerical and cat. data
```python
fig = px.pie(df, names = 'churn', template = 'simple_white', title = 'Churn')
fig.update_traces(rotation = 90, pull = [0.1], textinfo = 'percent+label')
fig.show()
```
![output_8_0](https://github.com/user-attachments/assets/093b916f-4f54-462c-b662-17aaad4b62d9)
```python
fig = px.pie(df, names = 'gender', template = 'simple_white', title = 'Gender')
fig.update_traces(rotation = 90, pull = [0.1], textinfo = 'percent+label')
fig.show()
```
![output_9_0](https://github.com/user-attachments/assets/a6755b8f-39c9-4528-9545-5c7dd00308d1)

```python
fig = px.pie(df, names = 'seniorcitizen', template = 'simple_white', title = 'Senior Citizen')
fig.update_traces(rotation = 90, pull = [0.1], textinfo = 'percent+label')
fig.show()
```
![output_10_0](https://github.com/user-attachments/assets/27678c11-6f0d-4de0-9474-268e2b6b387e)

```python
fig = px.pie(df, names = 'dependents', template = 'simple_white', title = 'Dependents')
fig.update_traces(rotation = 90, pull = [0.1], textinfo = 'percent+label')
fig.show()
```
![output_11_0](https://github.com/user-attachments/assets/2d6001ed-4a88-4080-8ee4-15d2e6ed9cf7)

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
![output_12_0](https://github.com/user-attachments/assets/485cf07d-9d33-41be-84e2-2c91c21a40be)

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
![output_13_0](https://github.com/user-attachments/assets/02a546cd-32cd-4af9-ab80-b9e77a05687a)

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
![output_14_0](https://github.com/user-attachments/assets/020c7210-d27e-44bb-b756-5e5f33c6dfe2)

```python
sns.boxplot(x='churn', y='tenure', data=df)
plt.title('Churn Rate')
plt.xlabel('churn')
plt.ylabel('tenure')
plt.show()
```
![output_15_0](https://github.com/user-attachments/assets/bc7d15c6-ee7c-4a58-af62-8413360d63a7)

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
![output_16_0](https://github.com/user-attachments/assets/7dcee72b-1af1-482b-9f86-bd2ab87ff213)

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
![output_17_0](https://github.com/user-attachments/assets/a1993d67-ee6c-4090-8c33-f0507e8908ea)
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
![output_18_0](https://github.com/user-attachments/assets/0a71532d-fd89-4846-8f12-1e4980775729)
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
![output_19_0](https://github.com/user-attachments/assets/9cfd646f-f9b6-49fb-a79b-7d90b64a93eb)
```python
sns.boxplot(x='churn', y='monthlycharges', data=df)
plt.title('Churn Rate')
plt.xlabel('churn')
plt.ylabel('monthlycharges')
plt.show()
```
![output_20_0](https://github.com/user-attachments/assets/f0d89e17-674f-4f16-9fe9-b42f8064a97d)
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

**Total churn rate**
```python
# Count the number of customers who churned and who did not, and calculate the overall churn rate
churn_count = df['churn'].value_counts()
total_customers = churn_count.iloc[0] + churn_count.iloc[1]
churn_rate = (churn_count.iloc[1] / total_customers) * 100

# Print the churn rate as a formatted string with 3 decimal places
print('The overall churn rate in the company is: {:.3f}%'.format(churn_rate))
```
The overall churn rate in the company is: 26.537%

**Sort the pivot table by the 'Churn' column in descending order**
```python
contract_churn = pd.pivot_table(data=df, index='contract', values='churn', aggfunc=lambda x: x.map({'Yes':1, 'No':0}).mean())

contract_churn.sort_values('churn', ascending=False)
```
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



**Spliting categorical columns into Nominal and Binary columns**
```python
cat_cols = ["gender","seniorcitizen","dependents","phoneservice","multiplelines"
                    ,"internetservice","contract"]


num_cols = ["tenure","monthlycharges"]

target_col = 'churn'

nominal_cols = ['gender','internetservice','contract']

binary_cols = ['seniorcitizen','dependents','phoneservice','multiplelines','internetservice','churn_rate']
```

**Churn vs univariables as Demographics**
```python
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
    

**Correlations and Linearity**
```python
category_corr = df[['churn_rate', 'seniorcitizen', 'tenure', 'monthlycharges']].corr(method='pearson')

# Print the churn_corr DataFrame
category_corr
```
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

**Plotting the heatmap Correlation**
```python
plt.figure(figsize=(8, 6))
sns.heatmap(category_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()
```
![output_31_0](https://github.com/user-attachments/assets/7a6f1fa9-b35b-4cbe-af8a-10fdb4845adc)

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

**Key features indicating churn**
```python
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
    
**Feature importances plot**
```python
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
![output_37_0](https://github.com/user-attachments/assets/e4e543b2-0383-465a-b41c-2116ae664bf3)

**Strategies to retain customers**
```python
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

**Label Encoding to Numeric**
```python
lenc = LabelEncoder()
df2 = df_transformed.copy()
for i in df_transformed.columns:
    df2[i] = lenc.fit_transform(df2[i])
```

```python
df2.head()
```

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

**Plot histograms and calculate skewness and kurtosis**
```python
from scipy.stats import skew, kurtosis
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


![output_43_1](https://github.com/user-attachments/assets/4cdea856-e660-43af-b19e-55d5c28f411e)

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
![output_49_0](https://github.com/user-attachments/assets/60988388-1b98-422e-8d62-5da76d32a10f)
