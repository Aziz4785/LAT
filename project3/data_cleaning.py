import pandas as pd
import random

# Read the CSV file
df = pd.read_csv('training_data_raw.csv')



# Remove duplicates
df.drop_duplicates(inplace=True)

# df['price'] = df['price'].round(2)
# df['ratio'] = df['ratio'].round(2)



# Save the processed data to a new CSV file
df.to_csv('cleaned_training_data.csv', index=False)

print(f"Original shape: {pd.read_csv('training_data_raw.csv').shape}")
print(f"Processed shape: {df.shape}")
print(f"Number of rows with to_buy = 0: {(df['to_buy'] == 0).sum()}")
print(f"Number of rows with to_buy = 1: {(df['to_buy'] == 1).sum()}")