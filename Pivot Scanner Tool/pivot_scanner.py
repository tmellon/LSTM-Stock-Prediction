import math
import numpy as np
import pandas as pd
from datetime import datetime

class PivotScanner:
    def __init__(self, data):
        self.chartData = data
        print(self.chartData)
        self.bars = []
        self.time = []
        self.curr_time = 0

        self.depth = 0
        self.maxHours = 0
        self.minPivots = 0
        self.dev_threshold = 0

        self.pivotCount = 0
        self.zig_zag_array = []
        self.pivot_times = []

        self.iLast = 0
        self.pLast = 0.0

        self.isHighLast = True
        self.linesCount = 0
        self.sumVol = 0
        self.sumVolLast = 0
        self.bar_index = 0

        self.lineLast = None
        self.curr_index = 0

    def initialize(self, depth, maxHours, minPivots, dev_threshold):
        self.depth = depth
        self.maxHours = maxHours
        self.minPivots = minPivots
        self.dev_threshold = dev_threshold

        self.pivotCount = 0
        self.zig_zag_array = []
        self.pivot_times = []

        self.iLast = 0
        self.pLast = 0.0

        self.isHighLast = True
        self.linesCount = 0
        self.sumVol = 0
        self.sumVolLast = 0
        self.bar_index = 0

        self.lineLast = None

    def datetime_to_ms(self, time):
        time = time[:19]
        dt_obj = datetime.strptime(time,
                                '%Y-%m-%d %H:%M:%S')
        millisec = dt_obj.timestamp() * 1000
        #print(millisec)
        return millisec

    def pivots(self, src, length, isHigh):
        if self.curr_index - length >= 0:
            p = self.chartData.iloc[self.curr_index - length][src]
        else:
            p = 0

        if length == 0:
            return self.datetime_to_ms(self.chartData.iloc[self.curr_index]['Datetime']), p
        else:
            isFound = True
            for i in range(abs(length - 1)):
                if isHigh == True and self.chartData.iloc[self.curr_index - i][src] > p:
                    isFound = False
                if isHigh == False and self.chartData.iloc[self.curr_index - i][src] < p:
                    isFound = False
            for i in range(length + 1, 2 * length):
                if isHigh == True and self.chartData.iloc[self.curr_index - i][src] >= p:
                    isFound = False
                if isHigh == False and self.chartData.iloc[self.curr_index - i][src] <= p:
                    isFound = False
            if isFound and length * 2 <= self.curr_index:
                return self.datetime_to_ms(self.chartData.iloc[self.curr_index - length]['Datetime']), p
            else:
                return None, None
    
    def calc_dev(self, base_price, price):
        if base_price == 0:
            return 0
        return 100 * (price - base_price) / base_price

    def pivotFound(self, dev, isHigh, index, price):
        if self.isHighLast == isHigh and self.lineLast != None:
            if (price > self.pLast if self.isHighLast else price < self.pLast):
                if self.linesCount <= 1:
                    self.lineLast[0] = (index, price)
                self.lineLast[1] = (index, price)
                return self.lineLast, self.isHighLast, False, self.sumVol + self.sumVolLast
            else:
                return None, None, False, None
        else:
            if self.lineLast == None:
                id = [(index, price), (index, price)]
                return id, isHigh, True, self.sumVol
            else:
                if abs(dev) >= self.dev_threshold:
                    id = [(self.iLast, self.pLast), (index, price)]
                    return id, isHigh, True, self.sumVol
                else:
                    return None, None, False, None
                
    def getTimeOfPivot5Ago(self):
        if len(self.pivot_times) >= self.minPivots:
            return self.pivot_times[len(self.pivot_times) - self.minPivots]
        else:
            return None
    
    def find_noise(self):
        iH = None
        iL = None
        result = []
        volatile = []

        print('Finding noise with hyperparameters:')
        print('Deviation (%):', self.dev_threshold)
        print('Depth:', self.depth)
        print('Max Hours', self.maxHours)
        print('Minimum Pivots', self.minPivots)

        for i in range(len(self.chartData)):
            #print(self.chartData.iloc[i]['Datetime'], self.datetime_to_ms(self.chartData.iloc[i]['Datetime']))
            self.bar_index = i
            time = i
            self.curr_time = time
            self.curr_index = i

            iH, pH = self.pivots(src='High', length=math.floor(self.depth / 2), isHigh=True)
            iL, pL = self.pivots(src='Low', length=math.floor(self.depth / 2), isHigh=True)

            if iH != None and iL != None and iH == iL:
                
                dev1 = self.calc_dev(self.pLast, pH)
                id2, isHigh2, isNew2, sum2 = self.pivotFound(dev=dev1, isHigh=True, index=iH, price=pH)
                if isNew2:
                    self.linesCount += 1
                    self.pivot_times.append(self.datetime_to_ms(self.chartData.iloc[i]['Datetime']))
                if id2 != None:
                    self.lineLast = id2
                    self.isHighLast = isHigh2
                    self.iLast = iH
                    self.pLast = pH
            
                dev2 = self.calc_dev(self.pLast, pL)
                id1, isHigh1, isNew1, sum1 = self.pivotFound(dev=dev2, isHigh=False, index=iL, price=pL)
                if isNew1:
                    self.linesCount += 1
                    self.pivot_times.append(self.datetime_to_ms(self.chartData.iloc[i]['Datetime']))
                if id1 != None:
                    self.lineLast = id1
                    self.isHighLast = isHigh1
                    self.iLast = iL
                    self.pLast = pL
            else:
                if iH != None:
                    dev1 = self.calc_dev(self.pLast, pH)
                    id, isHigh, isNew, sum = self.pivotFound(dev1, True, iH, pH)
                    if isNew:
                        self.linesCount += 1
                        self.pivot_times.append(self.datetime_to_ms(self.chartData.iloc[i]['Datetime']))
                    if id != None:
                        self.lineLast = id
                        self.isHighLast = isHigh
                        self.iLast = iH
                        self.pLast = pH
                else:
                    if iL != None:
                        dev2 = self.calc_dev(self.pLast, pL)
                        id, isHigh, isNew, sum = self.pivotFound(dev2, False, iL, pL)
                        if isNew:
                            self.linesCount += 1
                            self.pivot_times.append(self.datetime_to_ms(self.chartData.iloc[i]['Datetime']))
                        if id != None:
                            self.lineLast = id
                            self.isHighLast = isHigh
                            self.iLast = iL
                            self.pLast = pL

            #print(self.linesCount)
            time_of_pivot_5_ago = self.getTimeOfPivot5Ago()
            time_difference = 0
            if time_of_pivot_5_ago != None:
                # Get delta time from first pivot to now
                time_difference = self.datetime_to_ms(self.chartData.iloc[i]['Datetime']) - time_of_pivot_5_ago
            # Amount of time that has passed since first pivot has happened after reaching minimum
            hours_elapsed = time_difference / (60 * 60 * 1000)

            # if hours elapsed from first pivot to now after reaching minimum pivots is less than maxHours, shade current candle background
            # shade current candle background means add current candle to array of volatile
            #bgcolor(((shadeVolatile and hours_elapsed<maxHours) or ((not shadeVolatile) and hours_elapsed>maxHours)) and system1?color.blue:na,transp=90)

            if hours_elapsed < self.maxHours:
                #take everything between time_of_pivot_5_ago and current time
                #print(hours_elapsed)
                volatile.append(self.chartData.iloc[i].values)
                #print(volatile)
            else:
                if len(volatile) > 0:
                    #print(volatile)
                    #print('==========================================================\n\n')
                    result.append(volatile[:])
                    volatile.clear()
            

        return result
    
    # Package data as array of array of shaded period candle highs and lows
    def package_data():
        return 1
    
def main():
    data = pd.read_csv('./Pivot Scanner Tool/Market Data/EURUSD=X_start_date_2024-02-05_5m.csv')
    print(data.iloc[0].values, data.iloc[1].values)
    pv = PivotScanner(data)
    #pv.initialize(depth=4, maxHours=72, minPivots=3, dev_threshold=0.5)
    pv.initialize(depth=10, maxHours=58, minPivots=9, dev_threshold=0.23888)
    volatile = pv.find_noise()
    print(volatile)
    #print(volatile[0])
    #print(volatile[0][0])
    #print(len(volatile))
    #for i in volatile:
    #    #print(i)
    #    print(i[0][0], ' - ', i[len(i) - 1][0])


    print('Done!')

if __name__ == "__main__":
    main()