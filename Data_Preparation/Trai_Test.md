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

# Display the first 15 rows of the encoded DataFrame
df_encoded.head()
```