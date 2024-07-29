
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
df = pd.read_csv('Dataset(ATS).csv')
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