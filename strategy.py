#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
import warnings

warnings.filterwarnings("ignore")


# In[2]:


# This file is for the strategy
# strategy：inputs are the ticker symbols and the start day, the outputs are the positions of the stocks for each date 
# plot：inputs are ticker symbols and the dates, outputs are the graph of the close and the performance of the strategy 
# test：the input is the strategy, the output is its performance
# save：save the result in a pdf file


# In[3]:


# Build the MACD strategy:

def calculate_macd(data, short_period=12, long_period=26, signal_period=9):
    data['EMA_short'] = data['Adj Close'].ewm(span=short_period,
                                          adjust=False).mean()
    data['EMA_long'] = data['Adj Close'].ewm(span=long_period, adjust=False).mean()
    data['MACD'] = data['EMA_short'] - data['EMA_long']
    data['Signal_Line'] = data['MACD'].ewm(span=signal_period,
                                           adjust=False).mean()
    return data


def generate_signals(data):
    data['Signal'] = 0
    data['Signal'] = np.where(data['MACD'] > data['Signal_Line'], 1, 0)
    data['Position'] = data['Signal'].diff()
    return data['Position']


# In[4]:


def MACD(stocks, start_date, end_date):
    data = yf.download(stocks, start=start_date, end = end_date, progress = False)
    close = data['Adj Close']
    macd_result = pd.DataFrame(0, index=close.index, columns=close.columns)
    for stock in stocks:
        data = yf.download(stock, start=start_date, end = end_date, progress = False)
        data = calculate_macd(data)
        signal = generate_signals(data)
        macd_result[stock] = signal
    
    return macd_result


# In[5]:


# Test
if __name__ == '__main__':
    stocks = [
        'MMM', 'AXP', 'AMZN', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS'
    ]

    start_date = '2023-01-01'
    end_date = '2024-04-26'

    print(MACD(stocks, start_date, end_date))

