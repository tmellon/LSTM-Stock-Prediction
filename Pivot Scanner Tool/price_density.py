import pandas as pd

def price_density(result):
    pd_sum = 0
    #print(result)
    for res in result:
        higharr = []
        lowarr = []
        sumHL = 0

        for candle in res:
            high = candle[2]
            low = candle[3]
            higharr.append(high)
            lowarr.append(low)

            sumHL += high - low
        
        pd = sumHL / (max(higharr) - min(lowarr))
        pd_sum += pd

    mean_pd = pd_sum / len(result)
    #print('Mean Price Density:', mean_pd)
    #print(len(result))
    #print()
    return mean_pd

def mean_price_density(data, period):
        high = []
        low = []
        ma = []
        pdarr = []
        for i in range(period):
            high.append(data.iloc[i]['High'])
            low.append(data.iloc[i]['Low'])

        numerator = 0
        for i in range(period):
            numerator += high[i] - low[i]
        pmax = max(high)
        pmin = min(low)
        denominator = pmax - pmin

        print(numerator, denominator)

        pd_sum = numerator / denominator

        pdarr.append(pd_sum)
        ma.append(pd_sum)

        n = period + 1
        count = 1
        while n < len(data):
            newHigh = data.iloc[n]['High']
            newLow = data.iloc[n]['Low']

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
        #print(count)
        return meanPD