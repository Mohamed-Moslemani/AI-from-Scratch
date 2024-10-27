import math 
import scipy
import numpy as np 
import pandas as pd 

def SinglegradientDescent(x,y):
    w = 0
    b = 0
    n = len(x)
    iters = int(input("Enter number of max iterations: "))
    learning_rate = 0.01
    for i in range(iters):
        for j in range(len(x)):
            w = w - learning_rate * (1/n)*(w*x[j] + b - y[j])*x[j]
            b = b - learning_rate * (1/n)*(w*x[j] + b - y[j])
    
    print(w)
    print(b)


def MultipleGradientDescent(df,target):
    m,n = df.shape
    w = [0]*n
    x = df.drop(columns=[target])
    y = df[target]   
    lr = 0.01
    iters = 1000
    b = 0 
    for i in range(iters): 
        for j in range(m):
            for k in range(n):
                w[k] = w[k] + (lr/m)*(y.iloc[j] - w[k]*x.loc[j, x.columns[k]] - b) * x.loc[j, x.columns[k]]
                b = b + (lr/n)*(y[j] - w[k]*x.loc[j, x.columns[k]] - b)
    
    print(f"weights: {w}")
    print(f'bias: {b}')

data = {
    'name': [3, 433, 53],
    'age': [24, 25, 23],
    'score': [85, 88, 90]
}

small_df = pd.DataFrame(data)

MultipleGradientDescent(small_df,'age')