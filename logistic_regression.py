import numpy as np 
import pandas as pd 
import math 


def sigmoid(x,w,b):

    return 1/(1+np.exp(-(np.dot(w,x)+b)))

def logisticGradient(df,iters,learning_rate,target):

    m,n = df.shape
    x = df.drop(columns=[target]).values
    y= df[target].values
    w = np.zeros(n-1)
    b= 0 
    
    for i in range(iters):
        y_pred = sigmoid(w,x,b)
        dw = np.zeros(n-1)
        db = 0 

        for j in range(n-1):
            dw = sum([y_pred - y]*x[:,j])
        db = sum([y_pred-y])
    
        b-= learning_rate*db 
        w -= learning_rate*dw 

    return w,b 

def logisticClassifier(x,w,b):
    preds = []
    if sigmoid(x,w,b) < 0.5: 
        preds.append(0)
    else:
        preds.append(1)
    
    return preds


    
