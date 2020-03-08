import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import itemfreq

import sys
def meal(label):
    np.set_printoptions(threshold=sys.maxsize)

    df = pd.read_csv("MealNoMealData/"+label+".csv", usecols = [i for i in range(30)], header = None) 
    df = df.dropna(how='any') 


    a = np.zeros(shape=(len(df.index),29))
    max_value = 0
    dt = []
    dc = []
    for i in range(len(df.index)):
        sum = 0
        row = []
        for j in range(29):
            row.append(abs(df.iloc[i,j] - df.iloc[i,j+1]))
            sum +=  abs(df.iloc[i,j] - df.iloc[i,j+1])
        dt.append(sum)
        hist, _ = np.histogram(row, bins=3, range=(0,9), density=True)
        dc.append(hist)
        #a[i] = row
    return df
mealData = meal('mealData1')
nomealData = meal('Nomeal1')

a = []
b = []
#for i in range(len(mealData)):
    # a = itemfreq(mealData[i])
    # b = itemfreq(nomealData[i])
    # print("Meal Data", mealData[i])
    # print("No meal", nomealData[i])
    


#print(itemfreq(b))

x = np.arange(30)
y = mealData.iloc[5, 0:30]
plt.plot(x,y)
plt.show()

# b =np.array([[1,2]])
# c = np.array([[3,4]])
# a = np.matrix(b)
# print(a)
# a = np.concatenate((a,c))
# print(np.concatenate((a,c)))
# print(a[0][0])
