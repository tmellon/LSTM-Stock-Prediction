from numpy.lib import emath
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class QMCDetection:
    def __init__(self, data):
        self.data, self.data_all = self.load_data(data['ticker'], data['start_date'], data['end_date'])
        self.smoothCandles = []
        self.qmcPattern = []
        self.componentStart = 0
        self.componentEnd = 0

        self.qmcFormula = {
            'absMin': tuple,
            'L': tuple,
            'H': tuple,
            'LL': tuple,
            'HH': tuple
        }

    def load_data(self, ticker, start_date, end_date):
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        stock_data['Date'] = stock_data.index
        stock_data.reset_index(drop=True, inplace=True)
        print(stock_data)

        data_np = stock_data.to_numpy()

        return data_np, stock_data

    def simulateDataStream(self):
        return True

    def findQMCComponent(self, t):
        self.qmcPattern.clear()
        
        component = []
        allPrices = []

        i = t + 1
        index = -1
        isComponent = False

        while i < len(self.data):
            y = self.smoothCandle(self.data[i])
            x = self.data[i][6]
            allPrices.append((x, y))
            index += 1

            if len(allPrices) == 2:
                # Find L
                if allPrices[1][1] <= allPrices[0][1]:
                    allPrices.pop(0)
                    index = 0
                else:
                    component.append((x, self.data[i][2]))
            elif len(allPrices) > 2:
                a = allPrices[index - 2]
                b = allPrices[index - 1]
                c = allPrices[index]
                if (a[1] > b[1] < c[1]): #Find LL
                    component.append((b[0], self.data[i][2]))
                elif (a[1] < b[1] > c[1]):
                    component.append((b[0], self.data[i][1]))

            if len(component) == 3:
                isComponent = self.validateComponent(component)
            
                if isComponent:
                    self.componentStart = i - t - len(allPrices)
                    self.componentEnd = i - t
                    self.qmcPattern = allPrices
                    print(self.qmcFormula)
                    return {'Code': 1, 'Message': 'Valid QMC Component Found', 'End': self.componentEnd}
                else:
                    component.clear()
                    allPrices.clear()
                    i -= 2
                    index = -1

            i += 1
        return {'Code': -1, 'Message': 'No Valid QMC Components Detected in Time Period ' + str(t) + ' to ' + str(len(self.data)), 'End': i}
    
    from datetime import datetime
    def datetime_to_ms(self, time):
        dt_obj = datetime.strptime('20-12.2016 09:38:42,76',
                                '%d-%m-%Y %H-%M-%S,%f')
        millisec = dt_obj.timestamp() * 1000
    
    def validateComponent(self, component):
        if component[0][1] > component[2][1]:
            self.component = component
            self.qmcFormula['L'] = component[0]
            self.qmcFormula['H']= component[1]
            self.qmcFormula['LL'] = component[2]
            return True
        
        return False
        
    def findAbsMin(self, range):
        i = self.componentStart
        while i >= self.componentStart - range:
            y = self.smoothCandle(self.data[i])
            x = self.data[i][6]
            self.qmcPattern.insert(0, (x, y))

            a = self.qmcPattern[2]
            b = self.qmcPattern[1]
            c = self.qmcPattern[0]
            if (a[1] > b[1] < c[1]):   #b is a minima or 
                if b[1] < self.qmcFormula['LL'][1]:
                    self.qmcFormula['absMin'] = (b[0], self.data[i][2])
                    self.componentStart = i
                    return {'Code': 1, 'Message': 'Absolute Minima FOUND within Lookback Range: QMC Candidate is Valid', 'QMC Components': self.qmcFormula, 'End': self.componentEnd}
            #elif (a[1] < b[1] > c[1]):
            #    self.allPrices.insert(0, b)
            i -= 1
        
        return {'Code': 0, 'Message': 'Absolute Minima NOT FOUND in Lookback Range ' + str(range) + ': QMC Candidate is Invalid', 'QMC Components': None, 'End': self.componentEnd}
    
    def findHH(self, range):
        t = self.componentEnd
        LL = self.qmcFormula['LL'][1]
        H = self.qmcFormula['H'][1]
        absMin = self.qmcFormula['absMin'][1]

        while t < len(self.data) + range and t < len(self.data):
            y = self.smoothCandle(self.data[t])
            x = self.data[t][6]
            self.qmcPattern.append((x, y))

            a = self.qmcPattern[-3]
            b = self.qmcPattern[-2]
            c = self.qmcPattern[-1]

            if (a[1] > b[1] < c[1]):   #b is a minima or 
                if b[1] < LL or b[1] < absMin:   #change later so b < absMin resets entire process from position of b
                    return {'Code': 2, 'Message': 'Low Less Than LL or Absolute Minimum Found', 'End': t}
            elif (a[1] < b[1] > c[1]):
                if b[1] > H:
                    self.qmcFormula['HH'] = (b[0], self.data[t - 1][2])
                    return {'Code': 3, 'Message':'HH FOUND in Lookforward Range: QMC Candidate is Valid', 'QMC': self.qmcPattern, 'QMC Components' : self.qmcFormula, 'End': t}
            t += 1
        
        if t == len(self.data):
            return {'Code': -1, 'Message': 'Reached end of Data stream looking for HH: QMC Candidate is Invalid', 'End': t}
        return {'Code': 0, 'Message': 'HH NOT FOUND in Lookforward Range ' + str(range) + ': QMC Candidate is Invalid', 'End': t}
    
    def smoothCandle(self, candle):
        currOpen = candle[0]
        currLow = candle[2]

        smoothCandle = abs(currOpen + currLow) / 2

        return smoothCandle
    
    def getSmoothCandles(self):
        smoothedCandles = []
        for i in range(len(self.data) - 1):
            curr = self.data[i]
            currOpen = curr[0]
            currLow = curr[2]

            smooth = abs(currOpen + currLow) / 2
            smoothedCandles.append(smooth)

        return smoothedCandles

from datetime import datetime

def main():
    data = {
        'ticker':       'EURUSD=X',
        'start_date':   '2020-03-10',
        'end_date':     '2022-11-11',
    }

    tolerence = 10

    qmc = QMCDetection(data)
    #smooth = qmc.getSmoothCandles()
    #maxima, minima = qmc.getMaximaMinima(smooth)
    #mx = qmc.getMaximaHighs(maxima)
    #mn = qmc.getMinimaLows(minima)
    t = 0
    while t < len(qmc.data):
        print(t)
        componentSearch = qmc.findQMCComponent(t)
        print(componentSearch['Message'])
        if componentSearch['Code'] == -1:
            t = componentSearch['End']
            continue
        
        absMinSearch = qmc.findAbsMin(200)
        print(absMinSearch['Message'])
        if absMinSearch['Code'] == 0:
            t = absMinSearch['End']
            continue
        
        absMin = absMinSearch['QMC Components']['absMin']
        LL = absMinSearch['QMC Components']['LL']
        L = absMinSearch['QMC Components']['L'][1]

        x1 = absMin[0].timestamp() * 1000
        y1 = absMin[1]

        x2 = LL[0].timestamp() * 1000
        y2 = LL[1]

        m = (y2 - y1) / (x2 - x1)
        x = (L - y1 + (m * x1)) / m

        lookForward = int((x - x1) * 1000) + tolerence
        HHSearch = qmc.findHH(lookForward)
        print(HHSearch['Message'], HHSearch['Code'])
        if HHSearch['Code'] == 3:
            li = []
            for i in HHSearch['QMC']:
                li.append(i[1])
            li2 = []
            for i in HHSearch['QMC Components']:
                li2.append(HHSearch['QMC Components'][i][1])
            plt.figure(figsize=(15, 8))
            plt.title('Smoothed Candles')
            plt.plot(li)
            plt.plot(li2, 'ro')
            plt.xlabel('Time')
            plt.ylabel('Smoothed Value')
            plt.savefig('qmc' + str(t) + '.png')


            plt.figure(figsize=(15, 8))
            plt.title('QMC Pattern via Components')
            li2 = []
            for i in HHSearch['QMC Components']:
                li2.append(HHSearch['QMC Components'][i][1])
            #lists = sorted(HHSearch['QMC Components'].items()) # sorted by key, return a list of tuples
            #x, y = zip(*lists) # unpack a list of pairs into two tuples
            plt.plot(li2)
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.savefig('qmc component' + str(t) + '.png')
            t = HHSearch['End']
        else:
            t = HHSearch['End']
    '''
    plt.figure(figsize=(15, 8))
    plt.title('Price Density (All Time)')
    plt.plot(pdarr)
    plt.plot(ma[-1])
    plt.xlabel('Period #')
    plt.ylabel('Price Density')
    plt.savefig('pdday.png')

    TODO:
    Convert smoothing to SMA length 2 using HLC3 as source. Modify code to use SMA value for finding low candle lows and high candle highs.
    Fix while loop to continue from point where component fails
    Check lows and highs using candle low and high
    '''


if __name__ == "__main__":
    main()