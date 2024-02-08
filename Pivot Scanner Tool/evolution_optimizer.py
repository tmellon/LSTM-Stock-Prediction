import pandas as pd
import random
import pivot_scanner

# evolution strategy (mu, lambda) of the ackley objective function
from numpy import asarray
from numpy import argsort
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed
import numpy as np

class EvolutionOptimizer:
    def __init__(self, data):
        self.chartData = data
        self.pv = pivot_scanner.PivotScanner(data)

    def price_density(self, result):
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
        print('Mean Price Density:', mean_pd)
        print(len(result))
        print()
        return mean_pd

    def in_bounds(self, candidate):
        # enumerate all dimensions of the point
        if 0.1 <= candidate['Deviation'] <= 100 and 1 <= candidate['Depth'] <= 100 \
                                        and 1 <= candidate['Max Hours'] < 100 and 1 <= candidate['Min Pivots'] <= 50:
            return True
        return False

    def es_comma(self, n_iter, step_size, mu, lam):
        best, best_eval = None, 0
        # calculate the number of children per parent
        n_children = int(lam / mu)
        # initial population
        population = list()
        for _ in range(lam):
            candidate = None
            while candidate is None or not self.in_bounds(candidate):
                candidate = {}
                candidate['Deviation'] = round(random.uniform(0, 1), 5)
                candidate['Depth'] = int(random.uniform(1, 10))
                candidate['Max Hours'] = int(random.uniform(1, 100))
                candidate['Min Pivots'] = int(random.uniform(1, 20))
            population.append(candidate)
        # perform the search
        for epoch in range(n_iter):
            result = []
            scores = []
            count = 0
            for c in population:
                count += 1
                print('=========================')
                print('=    Candidate', count, '    =')
                print('=========================')
                self.pv.initialize(depth=c['Depth'], maxHours=c['Max Hours'], minPivots=c['Min Pivots'], dev_threshold=c['Deviation'])
                res = self.pv.find_noise()
                #print(res)
                result.append(res)
                if len(res) == 0:
                    print('No noise found')
                    print()
                    scores.append(0)
                elif len(res) < 3:
                    print('Insufficient amount of noise found')
                    print()
                    scores.append(0)
                else:
                    scores.append(self.price_density(res))
            # evaluate fitness for the population
            #scores = [self.price_density(c) for c in result]
            # rank scores in descending order
            ranks = argsort(scores)
            ranks = np.flip(ranks)
            # select the indexes for the top mu ranked solutions
            selected = [i for i,_ in enumerate(ranks) if ranks[i] < mu]
            # create children from parents
            children = list()
            for i in selected:
                # check if this parent is the best solution ever seen
                if scores[i] > best_eval:
                    best, best_eval = population[i], scores[i]
                    print('%d, Best: f(%s) = %.5f' % (epoch, best, best_eval))
                # create children for parent
                for _ in range(n_children):
                    child = None
                    while child is None or not self.in_bounds(child):
                        child = {}
                        child['Deviation'] = round((population[i]['Deviation'] + randn(1) * 0.15)[0], 5)
                        child['Depth'] = int((population[i]['Depth'] + randn(1) * random.uniform(1, 10) * 0.1)[0])
                        child['Max Hours'] = int((population[i]['Max Hours'] + randn(1) * random.uniform(1, 10) * 0.5)[0])
                        child['Min Pivots'] = int((population[i]['Min Pivots'] + randn(1) * random.uniform(1, 10) * 0.5)[0])
                    children.append(child)
            # replace population with children
            population = children
        return [best, best_eval]


def main():
    data = pd.read_csv('./Pivot Scanner Tool/Market Data/EURUSD=X_start_date_2024-02-05_5m.csv')
    #pv = pivot_scanner.PivotScanner(data)
    dm = EvolutionOptimizer(data)
    # seed the pseudorandom number generator
    seed(1)
    # define the total iterations
    n_iter = 10
    # define the maximum step size
    step_size = 0.15
    # number of parents selected
    mu = 20
    # the number of children generated by parents
    lam = 100
    # perform the evolution strategy (mu, lambda) search
    best, score = dm.es_comma(n_iter, step_size, mu, lam)
    print('Done!')
    print('f(%s) = %f' % (best, score))

if __name__ == "__main__":
    main()