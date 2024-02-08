from numpy.lib import emath
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class Noise:
    def __init__(self, data):
        self.data, self.data_all = self.load_data(data['ticker'], data['start_date'], data['end_date'])
        print(self.data_all)

    
    def load_data(self, ticker, start_date, end_date):
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        #stock_data.insert(0, 'Index', range(0, len(stock_data)))
        stock_data['Date'] = stock_data.index
        stock_data.reset_index(drop=True, inplace=True)
        
        #stock_data.set_index()
        #stock_data.head()

        data_np = stock_data.to_numpy()
        print(len(data_np))
        #print(data_np)
        #print(data_np[0][1])

        #plt.figure(figsize=(15, 8))
        #plt.title('Stock Prices History')
        #plt.plot(stock_data['Close'])
        #plt.xlabel('Date')
        #plt.ylabel('Prices ($)')
        #plt.savefig('Original.png')
        return data_np, stock_data
    
    def priceDensity(self, period):
        high = []
        low = []
        ma = []
        pdarr = []
        for i in range(period):
            high.append(self.data[i][1])
            low.append(self.data[i][2])

        n = period - 1
        numerator = 0
        for i in range(n):
            numerator += high[i] - low[i]
        pmax = max(high)
        pmin = min(low)
        denominator = pmax - pmin

        print(numerator, denominator)

        pd_sum = numerator / denominator

        pdarr.append(pd_sum)
        ma.append(pd_sum)

        count = 1
        while n < len(self.data) - 1:
            newHigh = self.data[n][1]
            newLow = self.data[n][2]

            prev_high = high.pop(0)
            prev_low = low.pop(0)

            high.append(newHigh)
            low.append(newLow)

            pmax = max(high)
            pmin = min(low)

            numerator = numerator - (prev_high - prev_low) + (newHigh - newLow)
            denominator = pmax - pmin
            pd = numerator / denominator
            #print(pd)
            pd_sum += pd
            n += 1
            count += 1
            pdarr.append(pd)
            ma.append(pd_sum/count)
            
            #print(count, pd)

        meanPD = pd_sum / count
        for i in range(len(pdarr) - 1):
            if pdarr[i] > meanPD:
               print(self.data_all.at[i, 'Date'], pdarr[i])
        #print(count)
        return meanPD, ma, pdarr

def main():
    data = {
        'ticker':       'EURUSD=X',
        'start_date':   '2018-01-01',
        'end_date':     '2024-02-01',
    }

    noise = Noise(data)

    meanPD, ma, pdarr = noise.priceDensity(20)
    print(meanPD)
    plt.figure(figsize=(15, 8))
    plt.title('Price Density Moving Average (All Time)')
    plt.plot(ma)
    plt.xlabel('Period #')
    plt.ylabel('Average Price Density')
    plt.savefig('pdma.png')

    plt.figure(figsize=(15, 8))
    plt.title('Price Density (All Time)')
    plt.plot(pdarr)
    plt.plot(ma[-1])
    plt.xlabel('Period #')
    plt.ylabel('Price Density')
    plt.savefig('pdday.png')
    

if __name__ == "__main__":
    main()