#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# In[2]:


class DataManager:
    def __init__(self, filename="stock_data.csv", directory="results"):
        self.directory = directory
        self.filename = filename
        self.filepath = os.path.join(directory, filename)
        # make sure that the directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)

    def update_data(self, tickers, start_date, end_date=None):
        """
        get or update the data of the stocks, and save the data in a csv file.
        """
        data = yf.download(tickers, start=start_date, end=end_date)
        data = data['Adj Close']
        if data == None:
            print('cannot load data of tickers:', tickers)
        data.to_csv(self.filepath)
        
        print("Data updated and saved locally.")

    def load_data(self, tickers, start_date, end_date=None):
        data = pd.read_csv(self.filepath, index_col='Date', parse_dates=True)
        return data[tickers][start_date:end_date]


# In[3]:


if __name__ == '__main__':
    dm = DataManager()
    # update or download the data
    dm.update_data(tickers=["AAPL", "GOOGL", "JPM"], start_date="2022-01-01")
    # load the data of the stocks that we need
    df = dm.load_data(tickers=["AAPL", 'JPM'],
                      start_date="2022-01-01",
                      end_date="2023-04-01")
    print(df)


# In[4]:


import strategy


class Backtester:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.portfolio = {}
        self.cash = 1000000 # initial cash
        self.initial_cash = self.cash
        self.transaction_log = pd.DataFrame(
            columns=['Date', 'Ticker', 'Action', 'Price', 'Quantity']) # DataFrame to record transactions
        self.portfolio_value = pd.DataFrame(columns=['Total Value'])

    def run_backtest(self,
                     tickers,
                     start_date,
                     end_date,
                     trading_strategy='default'):
        # get stocks' price data from DataManager
        price_data = self.data_manager.load_data(tickers, start_date, end_date)
        # generate signals using MACD strategy
        signals = strategy.MACD(tickers, start_date, end_date)
        self.portfolio_value.loc[start_date, 'Total Value'] = self.cash

        if trading_strategy == 'default':
            for date in signals.index:
                # deal with signals indicating sell
                sells = signals.loc[date] == -1
                for ticker in sells[sells].index:
                    if ticker in self.portfolio and self.portfolio[ticker] > 0:
                        quantity = self.portfolio[ticker]
                        price = price_data.loc[date, ticker]
                        self.portfolio[ticker] -= quantity
                        self.cash += price * quantity
                        self.record_transaction(date, ticker, 'SELL', price,
                                                quantity)

                # deal with signals indicating buy
                buys = signals.loc[date] == 1
                if buys.any():
                    buy_tickers = buys[buys].index
                    for ticker in buy_tickers:
                        max_investment = min(self.initial_cash * 0.1,
                                             self.cash)
                        price = price_data.loc[date, ticker]
                        if max_investment == np.nan or price == np.nan:
                            print('None value in data:')
                            print('Ticker:', ticker)
                            print('Date:', date)
                            print('price:', price)
                            print('Cash:', self.cash)
                            print('skip processing')
                            continue
                        quantity = int(max_investment // price // 100 * 100)
                        if quantity > 0 and self.cash >= price * quantity:
                            self.portfolio[ticker] = self.portfolio.get(
                                ticker, 0) + quantity
                            self.cash -= price * quantity
                            self.record_transaction(date, ticker, 'BUY', price,
                                                    quantity)

                # update portfolio value
                total_value = self.cash + sum(
                    self.portfolio[ticker] * price_data.loc[date, ticker]
                    for ticker in self.portfolio)
                self.portfolio_value.loc[date, 'Total Value'] = total_value

        return self.portfolio_value, self.transaction_log

    def record_transaction(self, date, ticker, action, price, quantity):
        # record the transactions in the DataFrame
        self.transaction_log = self.transaction_log.append(
            {
                'Date': date,
                'Ticker': ticker,
                'Action': action,
                'Price': price,
                'Quantity': quantity
            },
            ignore_index=True)


# In[5]:


if __name__ == '__main__':
    dm = DataManager()
    backtester = Backtester(dm)
    portfolio_value, transactions = backtester.run_backtest(['AAPL', 'GOOGL'],
                                                            '2024-01-01',
                                                            '2024-04-26')
    print(portfolio_value)
    print(transactions)


# In[6]:


class PerformanceAnalyzer:
    def __init__(self, portfolio_values, benchmark_ticker="^GSPC"):
        self.portfolio_values = portfolio_values
        self.returns = self.calculate_returns()
        self.benchmark_returns = self.get_benchmark_returns(benchmark_ticker)
        self.risk_free_rate = self.get_risk_free_rate() / 252  # get daily risk free rate
        self.excess_returns = self.returns - self.risk_free_rate  # get excess return

    def calculate_returns(self):
        """calculate daily retrun"""
        return self.portfolio_values.pct_change().fillna(0)

    def get_risk_free_rate(self):
        """get risk free rate using yfinance (^IRX is the 1-month T-bill rate)"""
        ticker = "^IRX"
        bond = yf.Ticker(ticker)
        hist = bond.history(period="1d")  # get the most recent data
        last_close = hist['Close'].iloc[-1]
        return last_close / 100  

    def get_benchmark_returns(self, ticker, start_date=None, end_date=None):
        """get benchmark returns"""
        if start_date is None:
            start_date = self.portfolio_values.index[0]
        if end_date is None:
            end_date = self.portfolio_values.index[-1]
        index = yf.download(ticker, start=start_date, end=end_date)
        return index['Adj Close'].pct_change().fillna(0)

    def annualized_return(self):
        """calculate annualized returns"""
        cumulative_return = self.portfolio_values.iloc[
            -1] / self.portfolio_values.iloc[0] - 1
        num_years = len(self.portfolio_values) / 252
        return (1 + cumulative_return)**(1 / num_years) - 1

    def sharpe_ratio(self):
        """calculate the annual Sharpe ration"""
        mean_return = np.mean(self.excess_returns)
        std_return = np.std(self.excess_returns)
        return (mean_return / std_return) * np.sqrt(252)

    def max_drawdown(self):
        """calculate the maximum drawdown"""
        cumulative_max = self.portfolio_values.cummax()
        drawdown = (self.portfolio_values - cumulative_max) / cumulative_max
        return drawdown.min()

# In[7]:


if __name__ == '__main__':
    analyzer = PerformanceAnalyzer(portfolio_value)
    print("Annualized Return:", analyzer.annualized_return())
    print("Sharpe Ratio:", analyzer.sharpe_ratio())
    print("Max Drawdown:", analyzer.max_drawdown())


# In[8]:


class ResultSaver:
    def __init__(self, transactions, portfolio_values):
        self.transactions = transactions
        self.portfolio_values = portfolio_values

    def save_to_csv(self):
        """save recorded transactions and portfolio values in a csv file"""
        self.transactions.to_csv('results/transactions.csv', index=True)
        self.portfolio_values.to_csv('results/portfolio_values.csv',
                                     index=True)

    def get_benchmark_data(self, ticker="^GSPC"):
        """get market benchmark data, for example S&P 500"""
        start_date = self.transactions["Date"].iloc[0]
        end_date = self.transactions["Date"].iloc[-1]
        data = yf.download(ticker, start=start_date, end=end_date)
        if data == None:
            print('cannot get benchmark data of ticker:', ticker)
        return data['Adj Close']

    def plot_and_save_pdf(self):
        """plot the comparison between portfolio values and market benchmark, save the graph in a pdf file"""
        benchmark_values = self.get_benchmark_data(ticker="^GSPC")
        self.portfolio_values.index = pd.to_datetime(
            self.portfolio_values.index)
        benchmark_returns = benchmark_values.pct_change().fillna(0)
        portfolio_returns = self.portfolio_values.pct_change().fillna(0)
        cumulative_portfolio_returns = (1 + portfolio_returns).cumprod() - 1
        cumulative_benchmark_returns = (1 + benchmark_returns).cumprod() - 1
        with PdfPages('results/comparison_chart.pdf') as pdf:
            plt.figure(figsize=(10, 6))
            plt.plot(cumulative_portfolio_returns.index,
                     cumulative_portfolio_returns,
                     label='Portfolio Value')
            plt.plot(cumulative_benchmark_returns.index,
                     cumulative_benchmark_returns,
                     label='Market Benchmark')
            plt.title('Portfolio Value vs Market Benchmark')
            plt.xlabel('Date')
            plt.ylabel('Return')
            plt.legend()
            pdf.savefig()
            plt.close()


# In[9]:


if __name__ == '__main__':
    saver = ResultSaver(transactions, portfolio_value)
    saver.save_to_csv()
    saver.plot_and_save_pdf()

