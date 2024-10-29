import math 
import numpy as np 
import pandas as pd 

def sigmoid(w,x,b):

    return 1/(1+math.exp^(-(np.dot(w,x)+b)))

def logloss(real_vls,preds):
    y_real = np.array(real_vls)
    y_pred = np.array(preds)
    m = len(y_real)
    loss = -np.sum(y_real * np.log(y_pred) + (1 - y_real) * np.log(1 - y_pred)) / m
    return loss

def logistic_gradient(df,target):
    nrows,ncols = df.shape
    x = df.drop(columns=target)
    y = y[target]
    w = [0]*(ncols - 1)
    b = 0
    y_preds = sigmoid(w,x,b)
    log_loss = logloss(y,y_preds)
    
def logistic_regressor():
    pass