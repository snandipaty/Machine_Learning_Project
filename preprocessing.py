import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the CSV file into a pandas DataFrame
csv_file_path = 'mushrooms.csv'
df = pd.read_csv(csv_file_path)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Iterate over each column in the DataFrame
for column in df.columns:
    # Check if the column has object (string) dtype
    if df[column].dtype == 'object':
        # Use LabelEncoder to transform the string values into numerical values
        df[column] = label_encoder.fit_transform(df[column])

# Save the modified DataFrame back to a CSV file
output_csv_path = 'processed.csv'
df.to_csv(output_csv_path, index=False)
