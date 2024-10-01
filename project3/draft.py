import pandas as pd 
import yfinance as yf
import datetime as dt

def get_current_50W_SMA(stock):
    today = pd.Timestamp.now()
    end_date = today + pd.DateOffset(days=2)
    start_date = end_date - pd.DateOffset(weeks=60)

    hist_weeks = yf.download(stock, start=start_date.strftime('%Y-%m-%d'), end=end_date, interval='1wk')
    if hist_weeks.empty:
        print("hist_weeks is empty")
        return None, None
    hist_weeks = hist_weeks.copy()
    hist_weeks['SMA_50'] = hist_weeks['Close'].rolling(window=50, min_periods=1).mean()
    hist_weeks['SMA_50'] = hist_weeks['SMA_50'].round().astype(int)
    closest_date = pd.Timestamp(today.date())
    print("closest_date before loop: ",closest_date)
    max_offset= 2
    while max_offset >0 and closest_date not in hist_weeks.index and closest_date > hist_weeks.index[0]:
        closest_date -= dt.timedelta(days=1)
        max_offset -=1

    if max_offset<0:
        return None,None
    
    print("closest_date : ",closest_date)
    sma_value = hist_weeks.loc[closest_date, 'SMA_50'] if closest_date in hist_weeks.index else None
    return sma_value, hist_weeks['SMA_50']

# res,hist_data = get_current_50W_SMA('NXPI')
# print(res)
# print()
# print()
#print(hist_data)

def get_current_10D_SMA(stock):
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

res,hist_data = get_current_10D_SMA('GOOG')
print(res)