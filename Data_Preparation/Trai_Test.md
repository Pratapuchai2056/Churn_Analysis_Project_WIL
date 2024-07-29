# Training and Testing Sets

**1. Introduction**

We will start the approach via the usage of importing the essential libraries this is needed for facts manipulation and model validation.
```
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
```
**2. Encoding the records**

We may be use the label encoding to transform specific variables into the numeric format.
```
# Load dataset
df = pd.read_csv('Dataset(ATS).csv')

# Initialize the LabelEncoder
lenc = LabelEncoder()

# Make a copy of the DataFrame for encoding
df_encoded = df.copy()

# Apply label encoding to each column
for column in df_encoded.columns:
    if df_encoded[column].dtype == 'object':  # Apply encoding only to categorical columns
        df_encoded[column] = lenc.fit_transform(df_encoded[column])

df_encoded.head()
```
<div>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>29.85</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>34</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>56.95</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>53.85</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>42.30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>70.70</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

**3. Splitting the Dataset**

Now, we split the dataset into training and testing sets. This allows us to train the model on one subset and evaluate it on another.
```
# Define features and target variable
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Display the shapes of the training and testing sets
print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')
```
```
    X_train shape: (5634, 9)
    X_test shape: (1409, 9)
    y_train shape: (5634,)
    y_test shape: (1409,)
```
**4. Save the training and testing sets as CSV files**

Finally, we save the training and testing datasets as CSV files. This allows for easy access and sharing of the datasets.
```
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
```
**5. Conclusion**

We have successfully split the dataset into training and testing sets and saved them as CSV files. This step is critical for validating the performance of machine learning models, ensuring they can generalize to unseen data.
