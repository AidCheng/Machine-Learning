import numpy as np
import pandas as pd

data = pd.read_csv('/Users/aiden/Developer/Modelling/Linear_Regression/data/world-happiness-report-2017.csv')
label = 'Family'

processed_data = data[[label]].values
length = processed_data.shape[0]

if __name__ == "__main__":
    
    def test_fc():
        j = 0
        for i in range(10):
            i+=j
            yield i
            j+=1
            print(i,j)
        
    num = test_fc()
    for i in range(10):
        print(next(num))