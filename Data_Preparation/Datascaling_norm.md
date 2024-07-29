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
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SeniorCitizen</th>
      <th>tenure</th>
      <th>MonthlyCharges</th>
      <th>gender_Male</th>
      <th>Dependents_Yes</th>
      <th>PhoneService_Yes</th>
      <th>MultipleLines_Yes</th>
      <th>InternetService_Fiber optic</th>
      <th>Contract_One year</th>
      <th>Contract_Two year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.439916</td>
      <td>-1.277445</td>
      <td>-1.160323</td>
      <td>-1.009559</td>
      <td>-0.654012</td>
      <td>-3.054010</td>
      <td>-0.854176</td>
      <td>-0.885660</td>
      <td>-0.514249</td>
      <td>-0.562975</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.439916</td>
      <td>0.066327</td>
      <td>-0.259629</td>
      <td>0.990532</td>
      <td>-0.654012</td>
      <td>0.327438</td>
      <td>-0.854176</td>
      <td>-0.885660</td>
      <td>1.944582</td>
      <td>-0.562975</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.439916</td>
      <td>-1.236724</td>
      <td>-0.362660</td>
      <td>0.990532</td>
      <td>-0.654012</td>
      <td>0.327438</td>
      <td>-0.854176</td>
      <td>-0.885660</td>
      <td>-0.514249</td>
      <td>-0.562975</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.439916</td>
      <td>0.514251</td>
      <td>-0.746535</td>
      <td>0.990532</td>
      <td>-0.654012</td>
      <td>-3.054010</td>
      <td>-0.854176</td>
      <td>-0.885660</td>
      <td>1.944582</td>
      <td>-0.562975</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.439916</td>
      <td>-1.236724</td>
      <td>0.197365</td>
      <td>-1.009559</td>
      <td>-0.654012</td>
      <td>0.327438</td>
      <td>-0.854176</td>
      <td>1.129102</td>
      <td>-0.514249</td>
      <td>-0.562975</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.439916</td>
      <td>-0.992402</td>
      <td>1.159546</td>
      <td>-1.009559</td>
      <td>-0.654012</td>
      <td>0.327438</td>
      <td>1.170719</td>
      <td>1.129102</td>
      <td>-0.514249</td>
      <td>-0.562975</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.439916</td>
      <td>-0.422317</td>
      <td>0.808907</td>
      <td>0.990532</td>
      <td>1.529024</td>
      <td>0.327438</td>
      <td>1.170719</td>
      <td>1.129102</td>
      <td>-0.514249</td>
      <td>-0.562975</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-0.439916</td>
      <td>-0.910961</td>
      <td>-1.163647</td>
      <td>-1.009559</td>
      <td>-0.654012</td>
      <td>-3.054010</td>
      <td>-0.854176</td>
      <td>-0.885660</td>
      <td>-0.514249</td>
      <td>-0.562975</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-0.439916</td>
      <td>-0.177995</td>
      <td>1.330711</td>
      <td>-1.009559</td>
      <td>-0.654012</td>
      <td>0.327438</td>
      <td>1.170719</td>
      <td>1.129102</td>
      <td>-0.514249</td>
      <td>-0.562975</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.439916</td>
      <td>1.206498</td>
      <td>-0.286218</td>
      <td>0.990532</td>
      <td>1.529024</td>
      <td>0.327438</td>
      <td>-0.854176</td>
      <td>-0.885660</td>
      <td>1.944582</td>
      <td>-0.562975</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-0.439916</td>
      <td>-0.788800</td>
      <td>-0.492281</td>
      <td>0.990532</td>
      <td>1.529024</td>
      <td>0.327438</td>
      <td>-0.854176</td>
      <td>-0.885660</td>
      <td>-0.514249</td>
      <td>-0.562975</td>
    </tr>
  </tbody>
</table>
</div>

**4.Saving the Normalized Data:**
And the fourth step might be to keep the normalized information as a CSV document.
```
df_scaled_df.to_csv('df_scaled_df.csv', index=False)
```
**5. The Conclusion:**
In conclusion , the noted component above, we have successfully done scaling techniques to
normalize the information and feature and after that saved the normalized facts as a CSV
record. These steps are vital in enhancing the overall commonplace performance of the
machine getting to know models.
