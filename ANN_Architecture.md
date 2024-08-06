```python
## Predictive ANN Modelling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import clone_model
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score, roc_curve
```


```python
df = pd.read_csv('Dataset(ATS).csv')
```


```python
df.head()
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
  </tbody>
</table>
</div>




```python
#Checking the nulls if any
missing_testues = df.isnull().sum()
print("Missing values per column:\n", missing_testues)
```

    Missing values per column:
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
df_cleaned = df.copy()
```


```python
# Convert Categorical Features to Numeric
categorical_cols = ['gender', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'Contract', 'Churn']
encoder = OneHotEncoder(sparse_output=False)
encoded_data = encoder.fit_transform(df_cleaned[categorical_cols])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
df_encoded = pd.concat([df_cleaned.drop(columns=categorical_cols), encoded_df], axis=1)
```


```python
df_encoded.head()
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
      <th>SeniorCitizen</th>
      <th>tenure</th>
      <th>MonthlyCharges</th>
      <th>gender_Female</th>
      <th>gender_Male</th>
      <th>Dependents_No</th>
      <th>Dependents_Yes</th>
      <th>PhoneService_No</th>
      <th>PhoneService_Yes</th>
      <th>MultipleLines_No</th>
      <th>MultipleLines_Yes</th>
      <th>InternetService_DSL</th>
      <th>InternetService_Fiber optic</th>
      <th>Contract_Month-to-month</th>
      <th>Contract_One year</th>
      <th>Contract_Two year</th>
      <th>Churn_No</th>
      <th>Churn_Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>29.85</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>34</td>
      <td>56.95</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>53.85</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>45</td>
      <td>42.30</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>2</td>
      <td>70.70</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Scaling Numerical Features
numerical_cols = ['tenure', 'MonthlyCharges']
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_encoded[numerical_cols])
scaled_df = pd.DataFrame(scaled_data, columns=numerical_cols)
df_scaled = df_encoded.copy()
df_scaled[numerical_cols] = scaled_df
```


```python
df_scaled.head()
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
      <th>SeniorCitizen</th>
      <th>tenure</th>
      <th>MonthlyCharges</th>
      <th>gender_Female</th>
      <th>gender_Male</th>
      <th>Dependents_No</th>
      <th>Dependents_Yes</th>
      <th>PhoneService_No</th>
      <th>PhoneService_Yes</th>
      <th>MultipleLines_No</th>
      <th>MultipleLines_Yes</th>
      <th>InternetService_DSL</th>
      <th>InternetService_Fiber optic</th>
      <th>Contract_Month-to-month</th>
      <th>Contract_One year</th>
      <th>Contract_Two year</th>
      <th>Churn_No</th>
      <th>Churn_Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>-1.277445</td>
      <td>-1.160323</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.066327</td>
      <td>-0.259629</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>-1.236724</td>
      <td>-0.362660</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0.514251</td>
      <td>-0.746535</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>-1.236724</td>
      <td>0.197365</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Split Data into Training and Testing Sets
X = df_scaled.drop(columns=['Churn_No', 'Churn_Yes'])
y = df_scaled['Churn_Yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Display the shapes of the training and testing sets
print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')
```

    X_train shape: (5634, 16)
    X_test shape: (1409, 16)
    y_train shape: (5634,)
    y_test shape: (1409,)
    


```python
print(y_train.unique())  # Should print [0, 1]
print(y_train.dtypes)    # Should be int64
```

    [0. 1.]
    float64
    


```python
# Convert classes to numpy array
classes = np.array([0, 1])

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights_dict = dict(enumerate(class_weights))

print("Calculated Class Weights:", class_weights_dict)
```

    Calculated Class Weights: {0: 0.680763653939101, 1: 1.8830213903743316}
    


```python
# Define a simple Sequential model
model = Sequential([
    Input(shape=(X_train.shape[1],)),  # Ensure input shape matches X_train
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model without class weights
history = model.fit(X_train, y_train, 
                    epochs=20, 
                    batch_size=32, 
                    validation_split=0.2)  # Split training data for validation
```

    Epoch 1/20
    [1m141/141[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 4ms/step - accuracy: 0.7094 - loss: 0.5616 - val_accuracy: 0.7924 - val_loss: 0.4271
    Epoch 2/20
    [1m141/141[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 0.7944 - loss: 0.4394 - val_accuracy: 0.8004 - val_loss: 0.4170
    Epoch 3/20
    [1m141/141[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 0.7894 - loss: 0.4350 - val_accuracy: 0.7959 - val_loss: 0.4132
    Epoch 4/20
    [1m141/141[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - accuracy: 0.7879 - loss: 0.4387 - val_accuracy: 0.7977 - val_loss: 0.4130
    Epoch 5/20
    [1m141/141[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 0.7826 - loss: 0.4397 - val_accuracy: 0.7950 - val_loss: 0.4104
    Epoch 6/20
    [1m141/141[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - accuracy: 0.7908 - loss: 0.4450 - val_accuracy: 0.7950 - val_loss: 0.4088
    Epoch 7/20
    [1m141/141[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - accuracy: 0.7954 - loss: 0.4228 - val_accuracy: 0.8039 - val_loss: 0.4109
    Epoch 8/20
    [1m141/141[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.7942 - loss: 0.4336 - val_accuracy: 0.7941 - val_loss: 0.4083
    Epoch 9/20
    [1m141/141[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 0.7944 - loss: 0.4246 - val_accuracy: 0.7950 - val_loss: 0.4086
    Epoch 10/20
    [1m141/141[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 0.7921 - loss: 0.4311 - val_accuracy: 0.7995 - val_loss: 0.4091
    Epoch 11/20
    [1m141/141[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 0.7959 - loss: 0.4273 - val_accuracy: 0.7968 - val_loss: 0.4088
    Epoch 12/20
    [1m141/141[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 0.7893 - loss: 0.4319 - val_accuracy: 0.7995 - val_loss: 0.4177
    Epoch 13/20
    [1m141/141[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 0.7989 - loss: 0.4297 - val_accuracy: 0.8057 - val_loss: 0.4082
    Epoch 14/20
    [1m141/141[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - accuracy: 0.7918 - loss: 0.4353 - val_accuracy: 0.8066 - val_loss: 0.4073
    Epoch 15/20
    [1m141/141[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 0.8082 - loss: 0.4181 - val_accuracy: 0.8004 - val_loss: 0.4077
    Epoch 16/20
    [1m141/141[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - accuracy: 0.8000 - loss: 0.4224 - val_accuracy: 0.7977 - val_loss: 0.4073
    Epoch 17/20
    [1m141/141[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - accuracy: 0.7863 - loss: 0.4380 - val_accuracy: 0.7977 - val_loss: 0.4090
    Epoch 18/20
    [1m141/141[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 0.8042 - loss: 0.4122 - val_accuracy: 0.8039 - val_loss: 0.4104
    Epoch 19/20
    [1m141/141[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 0.7934 - loss: 0.4206 - val_accuracy: 0.8039 - val_loss: 0.4098
    Epoch 20/20
    [1m141/141[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 0.7999 - loss: 0.4229 - val_accuracy: 0.7977 - val_loss: 0.4079
    


```python
# Evaluate on the training set
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Training Loss: {train_loss:.4f}")

# Evaluate on the validation set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"validation Accuracy: {test_accuracy:.4f}")
print(f"validation Loss: {test_loss:.4f}")
```

    Training Accuracy: 0.7973
    Training Loss: 0.4192
    validation Accuracy: 0.8119
    validation Loss: 0.4057
    


```python
# Plot training & validation accuracy testues
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'validation'], loc='upper left')
plt.show()

# Plot training & testidation loss testues
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'validation'], loc='upper left')
plt.show()
```


    
![png](output_14_0.png)
    



    
![png](output_14_1.png)
    



```python
# Convert y_test to NumPy array
y_test_array = np.array(y_test)

# Predict on the validation set
y_test_pred = (model.predict(X_test) > 0.5).astype("int32")

# Flatten the predictions and true labels
y_test_true = y_test_array.flatten()
y_test_pred = y_test_pred.flatten()


# Confusion Matrix
cm = confusion_matrix(y_test_true, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
report = classification_report(y_test_true, y_test_pred, target_names=['No Churn', 'Churn'])
print("Classification Report:")
print(report)
```

    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step
    


    
![png](output_15_1.png)
    


    Classification Report:
                  precision    recall  f1-score   support
    
        No Churn       0.86      0.89      0.87      1036
           Churn       0.67      0.58      0.62       373
    
        accuracy                           0.81      1409
       macro avg       0.76      0.74      0.75      1409
    weighted avg       0.81      0.81      0.81      1409
    
    


```python
# Saving the model
# model.save('model_trained_on_original_data.keras')
```


```python
# Clone the model architecture
model_copy = clone_model(model)

# Set the weights of the cloned model
model_copy.set_weights(model.get_weights())

# Compile the cloned model
model_copy.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Define the resampling strategy
smote = SMOTE(sampling_strategy='minority')
undersample = RandomUnderSampler(sampling_strategy='majority')

# Create a pipeline that first applies SMOTE, then undersampling
resampling_pipeline = Pipeline([
    ('smote', smote),
    ('undersample', undersample)
])

# Fit and transform the training data
X_resampled, y_resampled = resampling_pipeline.fit_resample(X_train, y_train)

# Train the cloned model with resampled data
history_resampled = model_copy.fit(X_resampled, y_resampled, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the cloned model
y_test_pred_proba = model_copy.predict(X_test).flatten()
```

    Epoch 1/20
    [1m207/207[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 3ms/step - accuracy: 0.7566 - loss: 0.4745 - val_accuracy: 0.6184 - val_loss: 0.6960
    Epoch 2/20
    [1m207/207[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - accuracy: 0.7646 - loss: 0.4725 - val_accuracy: 0.6492 - val_loss: 0.6577
    Epoch 3/20
    [1m207/207[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 0.7648 - loss: 0.4596 - val_accuracy: 0.7101 - val_loss: 0.6050
    Epoch 4/20
    [1m207/207[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 0.7659 - loss: 0.4739 - val_accuracy: 0.6751 - val_loss: 0.6330
    Epoch 5/20
    [1m207/207[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 0.7532 - loss: 0.4826 - val_accuracy: 0.6492 - val_loss: 0.6661
    Epoch 6/20
    [1m207/207[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.7676 - loss: 0.4713 - val_accuracy: 0.6842 - val_loss: 0.6274
    Epoch 7/20
    [1m207/207[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 4ms/step - accuracy: 0.7615 - loss: 0.4717 - val_accuracy: 0.6600 - val_loss: 0.6495
    Epoch 8/20
    [1m207/207[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 4ms/step - accuracy: 0.7696 - loss: 0.4670 - val_accuracy: 0.6359 - val_loss: 0.6691
    Epoch 9/20
    [1m207/207[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.7663 - loss: 0.4635 - val_accuracy: 0.7216 - val_loss: 0.5826
    Epoch 10/20
    [1m207/207[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - accuracy: 0.7733 - loss: 0.4581 - val_accuracy: 0.7180 - val_loss: 0.5709
    Epoch 11/20
    [1m207/207[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 0.7748 - loss: 0.4571 - val_accuracy: 0.6842 - val_loss: 0.6198
    Epoch 12/20
    [1m207/207[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 4ms/step - accuracy: 0.7676 - loss: 0.4746 - val_accuracy: 0.6999 - val_loss: 0.5930
    Epoch 13/20
    [1m207/207[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 4ms/step - accuracy: 0.7647 - loss: 0.4667 - val_accuracy: 0.6818 - val_loss: 0.6208
    Epoch 14/20
    [1m207/207[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.7656 - loss: 0.4661 - val_accuracy: 0.7156 - val_loss: 0.5756
    Epoch 15/20
    [1m207/207[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 0.7675 - loss: 0.4638 - val_accuracy: 0.6836 - val_loss: 0.6251
    Epoch 16/20
    [1m207/207[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 0.7788 - loss: 0.4558 - val_accuracy: 0.6618 - val_loss: 0.6292
    Epoch 17/20
    [1m207/207[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 0.7775 - loss: 0.4509 - val_accuracy: 0.7065 - val_loss: 0.5911
    Epoch 18/20
    [1m207/207[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.7694 - loss: 0.4662 - val_accuracy: 0.6256 - val_loss: 0.6945
    Epoch 19/20
    [1m207/207[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - accuracy: 0.7768 - loss: 0.4530 - val_accuracy: 0.6425 - val_loss: 0.6746
    Epoch 20/20
    [1m207/207[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 0.7723 - loss: 0.4593 - val_accuracy: 0.6244 - val_loss: 0.6805
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step
    


```python
# Compute ROC-AUC
roc_auc = roc_auc_score(y_test, y_test_pred_proba)
print(f"ROC-AUC (Resampled Data): {roc_auc:.4f}")

# Plot ROC Curve and Precision-Recall Curve
fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
plt.figure()
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Resampled Data)')
plt.show()
```

    ROC-AUC (Resampled Data): 0.8540
    


    
![png](output_18_1.png)
    



```python
# Compute Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_test_pred_proba)
average_precision = average_precision_score(y_test, y_test_pred_proba)
print(f"Average Precision (Resampled Data): {average_precision:.4f}")

plt.figure()
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Resampled Data)')
plt.show()
```

    Average Precision (Resampled Data): 0.6703
    


    
![png](output_19_1.png)
    



```python
# Predict on the test set using the cloned model
y_test_pred_resampled = (model_copy.predict(X_test) > 0.5).astype("int32").flatten()

# Compute and plot the Confusion Matrix
cm_resampled = confusion_matrix(y_test, y_test_pred_resampled)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_resampled, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Resampled Data)')
plt.show()

# Generate and print the Classification Report
report_resampled = classification_report(y_test, y_test_pred_resampled, target_names=['No Churn', 'Churn'])
print("Classification Report (Resampled Data):")
print(report_resampled)
```

    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    


    
![png](output_20_1.png)
    


    Classification Report (Resampled Data):
                  precision    recall  f1-score   support
    
        No Churn       0.87      0.86      0.86      1036
           Churn       0.62      0.65      0.63       373
    
        accuracy                           0.80      1409
       macro avg       0.74      0.75      0.75      1409
    weighted avg       0.80      0.80      0.80      1409
    
    


```python
# # Save the cloned model
# model_copy.save('model_trained_on_resampled_data.keras')
```


```python
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

def permutation_importance(model, X, y, metric, n_repeats=10):
    """
    Compute permutation importance for features.

    Parameters:
    - model: Trained model
    - X: Feature data
    - y: Target data
    - metric: Function to compute the metric (e.g., accuracy_score)
    - n_repeats: Number of repetitions for shuffling

    Returns:
    - importances: Dictionary of feature importances
    """
    # Ensure predictions are class labels
    y_pred = (model.predict(X) > 0.5).astype(int).flatten()
    baseline_score = metric(y, y_pred)
    importances = {}
    
    for feature in X.columns:
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            X_permuted[feature] = np.random.permutation(X_permuted[feature])
            y_pred_permuted = (model.predict(X_permuted) > 0.5).astype(int).flatten()
            permuted_score = metric(y, y_pred_permuted)
            scores.append(baseline_score - permuted_score)
        
        importances[feature] = np.mean(scores)
    
    return importances

# Calculate permutation importance
importances = permutation_importance(model, X_test, y_test, accuracy_score)

# Convert to DataFrame and sort
importance_df = pd.DataFrame(list(importances.items()), columns=['Feature', 'Importance'])
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print sorted DataFrame
print(importance_df)
```

    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 999us/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 893us/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 859us/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 904us/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 881us/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 911us/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 972us/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 711us/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 904us/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 720us/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 927us/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 779us/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 710us/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 711us/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 710us/step
                            Feature  Importance
    1                        tenure    0.069269
    11          InternetService_DSL    0.024840
    2                MonthlyCharges    0.021150
    15            Contract_Two year    0.019021
    3                 gender_Female    0.009084
    9              MultipleLines_No    0.007239
    13      Contract_Month-to-month    0.006813
    4                   gender_Male    0.004542
    7               PhoneService_No    0.004116
    8              PhoneService_Yes    0.003549
    12  InternetService_Fiber optic    0.002768
    14            Contract_One year    0.002271
    5                 Dependents_No    0.002129
    10            MultipleLines_Yes    0.001916
    6                Dependents_Yes    0.001419
    0                 SeniorCitizen    0.001065
    


```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot permutation importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Permutation Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
```


    
![png](output_23_0.png)
    



```python
#Comparision WHATIF
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Get feature importances from the trained model
importances = rf_model.feature_importances_

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance': importances
})

# Sort by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance from Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
```


    
![png](output_24_0.png)
    



```python
import pandas as pd
import numpy as np

# Predict churn probabilities
y_prob = model.predict(X_test).flatten()  # Flatten if necessary

# Define a threshold to classify churners vs non-churners
threshold = 0.5

# Classify based on the threshold
y_pred = (y_prob > threshold).astype(int)

# Convert to DataFrame for easy counting
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
    'Probability': y_prob
})

# Count actual churners and non-churners
actual_counts = results_df['Actual'].value_counts()
predicted_counts = results_df['Predicted'].value_counts()

print("Count of actual churners vs non-churners:")
print(actual_counts)

print("\nCount of predicted churners vs non-churners:")
print(predicted_counts)
```

    [1m45/45[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    Count of actual churners vs non-churners:
    Actual
    0.0    1036
    1.0     373
    Name: count, dtype: int64
    
    Count of predicted churners vs non-churners:
    Predicted
    0    1101
    1     308
    Name: count, dtype: int64
    


```python
print(f"Length of y_test: {len(y_test)}")
print(f"Length of y_pred: {len(y_pred)}")
import matplotlib.pyplot as plt

# Plot actual vs predicted counts
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Actual counts
ax[0].bar(actual_counts.index.astype(str), actual_counts.values, color='blue')
ax[0].set_title('Actual Churners vs Non-Churners')
ax[0].set_xlabel('Churn')
ax[0].set_ylabel('Count')

# Predicted counts
ax[1].bar(predicted_counts.index.astype(str), predicted_counts.values, color='red')
ax[1].set_title('Predicted Churners vs Non-Churners')
ax[1].set_xlabel('Churn')
ax[1].set_ylabel('Count')

plt.tight_layout()
plt.show()
```

    Length of y_test: 1409
    Length of y_pred: 1409
    


    
![png](output_26_1.png)
    



```python
import matplotlib.pyplot as plt

plt.hist(y_prob, bins=50, edgecolor='k')
plt.title('Distribution of Churn Probabilities')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.show()
```


    
![png](output_27_0.png)
    



```python
import matplotlib.pyplot as plt

# Assuming numerical_cols is a list of numerical column names
numerical_features = numerical_cols

# Plot histograms for numerical feature distributions
n_features = len(numerical_features)
fig, axs = plt.subplots(n_features, 1, figsize=(10, 5 * n_features), sharex=True)

for i, feature in enumerate(numerical_features):
    axs[i].hist(X_test[feature], bins=30, edgecolor='k', alpha=0.7)
    axs[i].set_title(f'Distribution of {feature}')
    axs[i].set_xlabel(feature)
    axs[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```


    
![png](output_28_0.png)
    

