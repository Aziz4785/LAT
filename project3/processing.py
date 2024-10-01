import pandas as pd
import yfinance as yf
import random
from datetime import datetime

def find_closest_previous_date(target_date, available_dates):
    target_date = pd.to_datetime(target_date)
    available_dates = pd.to_datetime(available_dates)
    previous_dates = available_dates[available_dates <= target_date]
    if not previous_dates.empty:
        return previous_dates.max()
    return None

def add_NonOpExp_column(df, Other_Non_Op_Inc_Exp_by_stock):
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Function to get gross profit for a row
    def get_NonOpExp(row):
        symbol = row['symbol']
        date = row['date']
        
        if symbol in Other_Non_Op_Inc_Exp_by_stock:
            profit_series = Other_Non_Op_Inc_Exp_by_stock[symbol]
            
            # Ensure the index is datetime
            profit_series.index = pd.to_datetime(profit_series.index)
            
            # Sort the series by date
            profit_series = profit_series.sort_index()
            
            
            closest_date = find_closest_previous_date(date, profit_series.index)
            if closest_date is not None:
                return profit_series[closest_date]
        
        return None

    # Apply the function to each row
    df['other_expense'] = df.apply(get_NonOpExp, axis=1)
    
    return df

def add_gross_profit_column(df, gross_profit_by_stock):

    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Function to get gross profit for a row
    def get_gross_profit(row):
        symbol = row['symbol']
        date = row['date']
        
        if symbol in gross_profit_by_stock:
            profit_series = gross_profit_by_stock[symbol]
            
            # Ensure the index is datetime
            profit_series.index = pd.to_datetime(profit_series.index)
            
            # Sort the series by date
            profit_series = profit_series.sort_index()
            
            
            closest_date = find_closest_previous_date(date, profit_series.index)
            if closest_date is not None:
                return profit_series[closest_date]
        
        return None

    # Apply the function to each row
    df['gross_profit'] = df.apply(get_gross_profit, axis=1)
    
    return df

df = pd.read_csv('cleaned_training_data.csv', parse_dates=['date'])



df['symbol'] = df['symbol'].str.upper()
start_date = df['date'].min() - pd.DateOffset(weeks=60)
min_date = df['date'].min()
start_date_10days = df['date'].min() - pd.DateOffset(days=20)
end_date = df['date'].max()
symbols = df['symbol'].unique()
sma_df = pd.DataFrame()

print("minimum date of the df is : ",df['date'].min())
print("so the start date of the hist value should be : ",start_date)


gross_profit_by_stock = {}
Other_Non_Op_Inc_Exp_by_stock = {}

for symbol in symbols:
    print(f"Processing symbol: {symbol}")
    if symbol == 'KSS':
        pd.set_option('display.max_rows', None)
    else:
        pd.set_option('display.max_rows', 20)
    #print(yf.Ticker(symbol).financials)
    #gross_profit_by_stock[symbol]=yf.Ticker(symbol).financials.loc['Gross Profit']
    Other_Non_Op_Inc_Exp_by_stock[symbol]=yf.Ticker(symbol).financials.loc['Net Non Operating Interest Income Expense']
    
    #10 days SMA
    hist_days = yf.download(symbol, start=start_date_10days.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval='1d')
    if hist_days.empty:
        print(f"No data fetched for {symbol}. Skipping.")
        continue
    hist_days = hist_days.copy()
    hist_days['SMA_10d'] = hist_days['Close'].rolling(window=10, min_periods=1).mean()
    hist_days = hist_days.loc[hist_days.index >= min_date]
    hist_days['SMA_10d'] = hist_days['SMA_10d'].round().astype(int)
    hist_days.reset_index(inplace=True)
    hist_days = hist_days[['Date','SMA_10d']].copy()
    hist_days['symbol'] = symbol
    #sma_df = pd.concat([sma_df, hist_days], ignore_index=True)

    hist_weeks = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval='1wk')
    if hist_weeks.empty:
        print(f"No data fetched for {symbol}. Skipping.")
        continue
    hist_weeks = hist_weeks.copy()
    # Calculate the 50-week SMA
    hist_weeks['SMA_50'] = hist_weeks['Close'].rolling(window=50, min_periods=1).mean()
    hist_weeks = hist_weeks.loc[hist_weeks.index >= min_date]
    hist_weeks['SMA_50'] = hist_weeks['SMA_50'].round().astype(int)
    # Reset index to access 'Date' column
    hist_weeks.reset_index(inplace=True)

    hist_weeks = hist_weeks[['Date', 'SMA_50']].copy()

    hist_weeks['symbol'] = symbol


    # Merge 10-day and 50-day SMAs
    symbol_sma = pd.merge(hist_days, hist_weeks, on=['Date', 'symbol'], how='outer')
    symbol_sma = symbol_sma.sort_values('Date')
    symbol_sma['SMA_50'] = symbol_sma.groupby('symbol')['SMA_50'].transform(lambda x: x.ffill())

    sma_df = pd.concat([sma_df, symbol_sma], ignore_index=True)


sma_df.rename(columns={'Date': 'date'}, inplace=True)


# Merge on 'date' and 'symbol'
merged_df = pd.merge(df, sma_df, on=['date', 'symbol'], how='left')

# If SMA_50 is NaN, fill it with the last available SMA value up to that date
#merged_df['SMA_50'] = merged_df.groupby('symbol')['SMA_50'].transform(lambda x: x.ffill())
merged_df['SMA_10d'] = merged_df.groupby('symbol')['SMA_10d'].transform(lambda x: x.ffill())


if 'close_price' in merged_df.columns:
    merged_df = merged_df.drop(columns=['close_price'])
merged_df = merged_df.dropna(subset=['SMA_50'])
merged_df = merged_df.dropna(subset=['SMA_10d'])

merged_df['price'] = merged_df['price'].round(2)

#merged_df=add_gross_profit_column(merged_df, gross_profit_by_stock)
merged_df=add_NonOpExp_column(merged_df, Other_Non_Op_Inc_Exp_by_stock)

if 'gross_profit' in merged_df.columns:
    merged_df['gross_profit'] = merged_df['gross_profit'] / 100000
merged_df['other_expense'] = merged_df['other_expense'] / 100


# Count the number of rows with to_buy = 0 and to_buy = 1
count_0 = (merged_df['to_buy'] == 0).sum()
count_1 = (merged_df['to_buy'] == 1).sum()
rows_to_remove = count_1 - count_0
if rows_to_remove > 0:
    indices_to_remove = merged_df[merged_df['to_buy'] == 1].index
    indices_to_remove = random.sample(list(indices_to_remove), rows_to_remove)
    merged_df = merged_df.drop(indices_to_remove)


merged_df.to_csv('processed_data.csv', index=False)
