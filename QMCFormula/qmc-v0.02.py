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
        self.qmcArr = []
        self.componentStart = 0
        self.componentEnd = 0

        self.raw_data_stream = []
        self.sma_2_hcl3 = []

        self.validQmcArr = []

    def load_data(self, ticker, start_date, end_date):
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        stock_data['Date'] = stock_data.index
        stock_data.reset_index(drop=True, inplace=True)
        print(stock_data)

        data_np = stock_data.to_numpy()

        return data_np, stock_data

    def simulateDataStream(self):
        mnArr = []
        mxArr = []
        mnMxArr = []

        absMinFound = False
        LFound = False
        HFound = False
        LLFound = False
        HHFound = False
        qmcFound = False

        absMin = ()
        L = ()
        H = ()
        LL = ()
        HH = ()

        # Will be while when using non static data stream
        # May have to change minmax comparison to use low price and high price in comparison
        for i in range(len(self.data)):
            candle = self.data[i]

            self.raw_data_stream.append({
                'Date': candle[6],
                'Open': candle[0],
                'High': candle[1],
                'Low': candle[2],
                'Close': candle[3]
            })

            if len(self.raw_data_stream) == 1:
                self.sma_2_hcl3.append((0, 0))
                continue
            
            sma2 = self.getSma2Hcl3(self.raw_data_stream[i - 1], self.raw_data_stream[i])
            self.sma_2_hcl3.append((i, sma2))

            if len(self.raw_data_stream) < 4:
                continue

            if not qmcFound:
                a = self.sma_2_hcl3[i - 2]
                b = self.sma_2_hcl3[i - 1]
                c = self.sma_2_hcl3[i]

                mxMn = self.isMaximaMinima(a[1], b[1], c[1])
                
                if mxMn == 0:
                    mnArr.append(b)
                    mnMxArr.append(b)
                elif mxMn == 1:
                    mxArr.append(b)
                    mnMxArr.append(b)
                if mxMn == 2:
                    continue

                if not absMinFound:
                    if mxMn == 0:
                        absMinFound = True
                        absMin = b
                    continue

                if not LFound:
                    if mxMn == 0:
                        if b[1] <= absMin[1]:     # Found L < absMin, L is new absMin
                            absMin = b
                        elif b[1] > absMin[1]:
                            L = b
                            LFound = True
                    continue

                if not HFound:
                    if mxMn == 1:
                        HFound = True
                        H = b
                    continue

                if not LLFound:
                    if mxMn == 0:
                        if b[1] <= absMin[1]:
                            absMin = b
                            LFound = False
                            HFound = False
                        elif b[1] >= L[1]:
                            L = b
                            HFound = False
                        elif absMin[1] < b[1] < L[1]:
                            LL = b
                            LLFound = True
                    continue

                if not HHFound:
                    if mxMn == 0:
                        if b[1] < LL[1] and b[1] > absMin[1]:
                            L = LL
                            H = mxArr[-2]
                            LL = b
                        if b[1] < absMin[1]:
                            absMin = b
                            LFound = False
                            HFound = False
                            LLFound = False
                    elif mxMn == 1:
                        if b[1] > H[1]:
                            HH = b
                            HHFound = True
                            qmcFound = True

            if qmcFound:
                self.validQmcArr.append({
                    '1': absMin,
                    'L': L,
                    'H': H,
                    'LL': LL,
                    'HH': HH,
                })
                absMin = ()
                L = ()
                H = ()
                LL = ()
                HH = ()

                absMinFound = False
                LFound = False
                HFound = False
                LLFound = False
                HHFound = False
                qmcFound = False

        return self.validQmcArr, self.raw_data_stream, self.sma_2_hcl3, mnMxArr
    
    # Increasing period used to calculate sma would cause triggers on only very clear entries (decrease qma's found)
    def getSma2Hcl3(self, c1, c2):
        c1_hcl3 = (c1['High'] + c1['Open']) / 2
        c2_hcl3 = (c2['High'] + c2['Open']) / 2

        sma2 = (c1_hcl3 + c2_hcl3) / 2

        return sma2
    
    def isMaximaMinima(self, a, b, c):
        if (a > b < c): # b is a minima
            return 0
        elif (a < b > c): # b is a maxima
            return 1
        
        return 2
    
    def verifyQMC(self, absMin, L, LL, HH):
        x1 = absMin[0]
        y1 = absMin[1]

        x2 = LL[0]
        y2 = LL[1]

        m = (y2 - y1) / (x2 - x1)
        x = (L[1] - y1 + (m * x1)) / m

        if HH[0] >= x:
            return False
        
        return True


from datetime import datetime

def main():
    data = {
        'ticker':       'EURUSD=X',
        'start_date':   '2020-03-10',
        'end_date':     '2022-11-11',
    }

    tolerence = 10

    qmc = QMCDetection(data)
    result, raw_datastream, sma_2_hcl3, mnMxArr = qmc.simulateDataStream()
    print(len(result))
    #smooth = qmc.getSmoothCandles()
    #maxima, minima = qmc.getMaximaMinima(smooth)
    #mx = qmc.getMaximaHighs(maxima)
    #mn = qmc.getMinimaLows(minima)

    plt.figure(figsize=(15, 8))
    plt.title('Data')
    plt.plot(*zip(*sma_2_hcl3[1:]))
    #plt.plot(*zip(*mnMxArr), 'ro')
    for qmc_data in result:
        qmc_plot = sma_2_hcl3[qmc_data['1'][0]: qmc_data['HH'][0] + 1]
        plt.plot(*zip(*qmc_plot))
    plt.xlabel('Period #')
    plt.ylabel('Price Density')
    plt.savefig('pdday.png')

    numValid = 0
    counter = 0
    for qmc_ in result:
        absMin = (qmc_['1'][0], raw_datastream[qmc_['1'][0]]['Low'])
        L = (qmc_['L'][0], raw_datastream[qmc_['L'][0]]['Low'])
        LL = (qmc_['L'][0], raw_datastream[qmc_['LL'][0]]['Low'])
        HH = (qmc_['L'][0], raw_datastream[qmc_['HH'][0]]['Low'])

        valid = qmc.verifyQMC(absMin, L, LL, HH)
        if valid:
            print(counter)
            numValid += 1
        counter += 1
    print(numValid)
    '''
    TODO:
    Convert smoothing to SMA length 2 using HLC3 as source. Modify code to use SMA value for finding low candle lows and high candle highs.
    Fix while loop to continue from point where component fails
    Check lows and highs using candle low and high

    Decide to open thread for trade execution calculation or use thread with pipeline to transmit qmc that have been found for trade entry calculation
    '''


if __name__ == "__main__":
    main()