# Scaling of the Data and the Normalization
**Introduction:**
The part below will briefly introduce us to tells the method or the ways that we need to follow the scaling techniques to normalize the statistics for reinforcing the model performance.
**1. Importing the Libraries:**
We will first start our preliminary manner by uploading the necessary libraries that is required statistics scaling.
```
From sklearn.Preprocessing import Standardscaler
```
**2. Data Scaling and Normalization:**
Secondly, we will now follow the Standard Scaler to normalize the given facts.
```
Scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_encoded)
```
**3. Converting the Normalized Data Back to DataFrame:**
Then, the third step would be to convert the normalized facts back to a Data Frame.
```
columns = ['SeniorCitizen', 	'tenure', 	'MonthlyCharges', 	'gender_Male', 	'Dependents_Yes', 	'PhoneService_Yes', 'MultipleLines_Yes', 	'InternetService_Fiber optic', 	'Contract_One year', 	'Contract_Two year']
df_scaled_df = pd.DataFrame(df_scaled, columns=columns)
df_scaled_df.head()
```

**4.Saving the Normalized Data:**
And the fourth step might be to keep the normalized information as a CSV document.
```
df_normalized.to_csv('df_normalized.csv', index=False)
```
**5. The Conclusion:**
In conclusion , the noted component above, we have successfully done scaling techniques to
normalize the information and feature and after that saved the normalized facts as a CSV
record. These steps are vital in enhancing the overall commonplace performance of the
machine getting to know models.
