import pandas as pd
import numpy as np
from datetime import datetime, timedelta



df = pd.read_csv('data/input_csvs/Products.csv')
df['ReleaseDate (D)'] = pd.to_datetime(df['ReleaseDate (D)'])

# Number of rows to add
num_new_rows = 1000000

# Generate new rows dynamically
last_id = df['Products_id (N) (P)'].max()

# Pre-generate data for better performance
new_ids = np.arange(last_id + 1, last_id + num_new_rows + 1)
new_names = [f"Product {chr(70 + (i % 26))}" for i in range(num_new_rows)]
new_prices = np.round(np.random.uniform(10, 100, num_new_rows), 2)
max_date = df['ReleaseDate (D)'].max()
new_dates = max_date + pd.to_timedelta(np.random.randint(30, 365, num_new_rows), unit='D')

# Create a DataFrame directly instead of stacking arrays with incompatible types
new_df = pd.DataFrame({
	'Products_id (N) (P)': new_ids,
	'ProductName (T)': new_names,
	'Price (N)': new_prices,
	'ReleaseDate (D)': new_dates
})

# Ensure the data types match the original DataFrame
for col in df.columns:
	new_df[col] = new_df[col].astype(df[col].dtype)
df = pd.concat([df, new_df], ignore_index=True)

# Display result
print(df)

df.to_csv('data/input_csvs/Products.csv', index=False)
