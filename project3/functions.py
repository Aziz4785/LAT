import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import random
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime as dt


#THIS FILE CONTAINS FUNCTIONS TO USE DURING SIMULATION AND DURING INFERENCE
"""
functions to use during simulation
"""
def get_OtherExpenses(stock_symbol):
    ticker = yf.Ticker(stock_symbol)
    financials = ticker.financials

    if 'Net Non Operating Interest Income Expense' in financials.index:
        revenue = financials.loc['Net Non Operating Interest Income Expense']
        return revenue
    else:
        return None
    
def get_recent_other_expenses(other_expenses_dict, stock, random_date):
    other_expenses = other_expenses_dict[stock]
    recent_other_exp_date = None
    recent_other_exp_value = None

    if other_expenses is not None:
        if not isinstance(other_expenses.index, pd.DatetimeIndex):
            other_expenses.index = pd.to_datetime(other_expenses.index)
        
        other_expenses = other_expenses.sort_index(ascending=True)
        recent_other_expenses = other_expenses.loc[:random_date]
        
        if not recent_other_expenses.empty:
            recent_other_exp_date = recent_other_expenses.index[-1]  # Get the last date from the index
            
            if recent_other_exp_date is not None and recent_other_exp_date >= random_date - timedelta(days=365):
                recent_other_exp_value = recent_other_expenses.iloc[-1]
        else:
            # Handle the case where `recent_other_expenses` is empty
            recent_other_exp_date = None
            recent_other_exp_value = None
    
    return recent_other_exp_date, recent_other_exp_value/100

def calculate_daily_moving_average(stock, random_date, num_periods=10):
    start_date = random_date - pd.DateOffset(days=20)
    end_date = random_date + pd.DateOffset(days=1)
    # Calculate start date to ensure enough data
    #start_date = random_date - dt.timedelta(days=int(num_periods*1.3))
    hist_weeks = yf.download(stock, start=start_date.strftime('%Y-%m-%d'), end=end_date, interval='1d')

    if hist_weeks.empty:
        return None, None
    
    hist_weeks = hist_weeks.copy()
    # Calculate 10 days SMA
    hist_weeks['SMA_10d'] = hist_weeks['Close'].rolling(window=10, min_periods=1).mean()

    # Filter data from the start date
    min_date = random_date - dt.timedelta(weeks=num_periods)
    hist_weeks = hist_weeks.loc[hist_weeks.index >= min_date]

    # Round SMA values to integers
    hist_weeks['SMA_10d'] = hist_weeks['SMA_10d'].round().astype(int)

    # Get the SMA value for the random date
    closest_date = pd.to_datetime(random_date)

    while closest_date not in hist_weeks.index and closest_date > hist_weeks.index[0]:
        closest_date -= dt.timedelta(days=1)

    sma_value = hist_weeks.loc[closest_date, 'SMA_10d'] if closest_date in hist_weeks.index else None
    
    return sma_value, hist_weeks['SMA_10d']


def calculate_weekly_moving_average(stock, random_date, num_periods=50):

    # Calculate start date to ensure enough data
    #start_date = random_date - dt.timedelta(weeks=int(num_periods*1.3))
    start_date = random_date - pd.DateOffset(weeks=60)
    end_date = random_date + pd.DateOffset(weeks=1)
    hist_weeks = yf.download(stock, start=start_date.strftime('%Y-%m-%d'), end=end_date, interval='1wk')

    if hist_weeks.empty:
        return None, None
    hist_weeks = hist_weeks.copy()
    # Calculate 50-week SMA
    hist_weeks['SMA_50'] = hist_weeks['Close'].rolling(window=50, min_periods=1).mean()

    # Filter data from the start date
    min_date = random_date - dt.timedelta(weeks=num_periods)
    hist_weeks = hist_weeks.loc[hist_weeks.index >= min_date]

    # Round SMA values to integers
    hist_weeks['SMA_50'] = hist_weeks['SMA_50'].round().astype(int)

    # Get the SMA value for the random date
    closest_date = pd.to_datetime(random_date)

    while closest_date not in hist_weeks.index and closest_date > hist_weeks.index[0]:
        closest_date -= dt.timedelta(days=1)

    sma_value = hist_weeks.loc[closest_date, 'SMA_50'] if closest_date in hist_weeks.index else None

    return sma_value, hist_weeks['SMA_50']

"""
functions to use during inference
"""
def get_current_price(stock):
    return 0

def get_current_50W_SMA(stock):
    today = pd.Timestamp.now()
    end_date = today + pd.DateOffset(days=2)
    start_date = end_date - pd.DateOffset(weeks=60)

    hist_weeks = yf.download(stock, start=start_date.strftime('%Y-%m-%d'), end=end_date, interval='1wk')
    if hist_weeks.empty:
        return None, None
    hist_weeks = hist_weeks.copy()
    hist_weeks['SMA_50'] = hist_weeks['Close'].rolling(window=50, min_periods=1).mean()
    hist_weeks['SMA_50'] = hist_weeks['SMA_50'].round().astype(int)
    closest_date = pd.Timestamp(today.date())

    max_offset= 2
    while max_offset >0 and closest_date not in hist_weeks.index and closest_date > hist_weeks.index[0]:
        closest_date -= dt.timedelta(days=1)
        max_offset -=1

    if max_offset<0:
        return None,None
    
    sma_value = hist_weeks.loc[closest_date, 'SMA_50'] if closest_date in hist_weeks.index else None
    return sma_value, hist_weeks['SMA_50']

def get_current_10d_SMA(stock):
    today = pd.Timestamp.now()
    end_date = today + pd.DateOffset(days=2)
    start_date = end_date - pd.DateOffset(days=20)

    hist_weeks = yf.download(stock, start=start_date.strftime('%Y-%m-%d'), end=end_date, interval='1d')
    if hist_weeks.empty:
        return None, None
    hist_weeks = hist_weeks.copy()
    hist_weeks['SMA_10d'] = hist_weeks['Close'].rolling(window=10, min_periods=1).mean()
    hist_weeks['SMA_10d'] = hist_weeks['SMA_10d'].round().astype(int)
    closest_date = pd.Timestamp(today.date())

    max_offset= 2
    while max_offset >0 and closest_date not in hist_weeks.index and closest_date > hist_weeks.index[0]:
        closest_date -= dt.timedelta(days=1)
        max_offset -=1

    if max_offset<0:
        return None,None
    
    sma_value = hist_weeks.loc[closest_date, 'SMA_10d'] if closest_date in hist_weeks.index else None
    return sma_value, hist_weeks['SMA_10d']

def get_current_other_expenses(other_expenses_dict,stock):
    today = pd.Timestamp.now().date()
    return get_recent_other_expenses(other_expenses_dict, stock, today)


"""
testing all these functions
"""
def compare_sma(calculated_sma, csv_sma,EPSILON):
    difference = calculated_sma - csv_sma
    if abs(difference)>EPSILON:
        return False
    return True

def compare_exp(calculated_sma, csv_sma,EPSILON):
    difference = calculated_sma - csv_sma
    if abs(difference)>EPSILON:
        return False
    return True

def test():
    EPSILON = 0.000004
    NBR_LOOP = 1500
    df = pd.read_csv('processed_data.csv',parse_dates=['date'])
    symbols = df['symbol'].unique()

    other_expenses_dict = {}
    for stock in symbols:
        otherexp = get_OtherExpenses(stock)
        other_expenses_dict[stock]=otherexp
    

    wrong_sma10 = 0
    wrong_sma50 = 0
    wrong_exp = 0
    for _ in range(NBR_LOOP):  
        random_row = df.sample(n=1).iloc[0]
        
        date = random_row['date']
        symbol = random_row['symbol']
        
        calculated_sma,hist_sma50 = calculate_weekly_moving_average(symbol, date)
        calculated_sma10,hist_sma10 = calculate_daily_moving_average(symbol, date)
        _,calculated_otherExp = get_recent_other_expenses(other_expenses_dict, symbol, date)

        csv_sma = random_row['SMA_50']
        csv_ma10 = random_row['SMA_10d']
        csv_otherExp = random_row['other_expense']

        if not compare_sma(calculated_sma, csv_sma,EPSILON)  :
            # print("problem in row : ")
            # print(random_row)
            # print(f"Calculated SMA50: {calculated_sma}")
            # print(f"CSV SMA_50: {csv_sma}")
            # print("historical values of the calculated SMA50 :")
            # print(hist_sma50)
            wrong_sma50+=1
        if not compare_sma(calculated_sma10, csv_ma10,EPSILON):
            # print("problem in row : ")
            # print(random_row)
            # print(f"Calculated SMA10: {calculated_sma10}")
            # print(f"CSV SMA_10: {csv_ma10}")
            # print("historical values of the calculated SMA10 :")
            # print(hist_sma10)
            wrong_sma10+=1
        if not compare_exp(calculated_otherExp,csv_otherExp,100):
            # print("problem in row : ")
            # print(random_row)
            # print(f"Calculated exp: {calculated_otherExp}")
            # print(f"CSV exp: {csv_otherExp}")
            wrong_exp+=1

    print(wrong_sma10/NBR_LOOP)
    print(wrong_sma50/NBR_LOOP)
    print(wrong_exp/NBR_LOOP)
if __name__ == "__main__":
    test()