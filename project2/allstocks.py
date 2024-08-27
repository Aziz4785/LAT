import os
import pandas as pd

# Define the folder containing the CSV files
folder_path = r'C:\Users\aziz8\Desktop\BASIC_strategies'
REMOVE_SUFFIX = False
# Initialize an empty DataFrame
all_tickers = pd.DataFrame()

source_files = ["CAC40.csv","Euronext.csv","FTSE.csv","sp_500_stocks.csv"]
# Loop through all the CSV files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv') and filename in source_files:
        print(filename)
        file_path = os.path.join(folder_path, filename)
        
        # Try reading the CSV with different encodings
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='ISO-8859-1')  # Common alternative encoding
            except UnicodeDecodeError:
                print(f"Failed to read {file_path} due to encoding issues.")
                continue  # Skip this file if it cannot be read

        # Check if the 'Ticker' column exists in the file
        if 'Ticker' in df.columns:
            if REMOVE_SUFFIX:
                df['Ticker'] = df['Ticker'].str.split('.').str[0]
            # Add the 'Ticker' column to the all_tickers DataFrame
            all_tickers = pd.concat([all_tickers, df[['Ticker']]], ignore_index=True)

# Drop duplicates
all_tickers.drop_duplicates(inplace=True)

# Save the result to a new CSV file
output_file = r'C:\Users\aziz8\Desktop\BASIC_strategies\allstocks.csv'
all_tickers.to_csv(output_file, index=False)

print(f"Combined CSV file saved to {output_file}")