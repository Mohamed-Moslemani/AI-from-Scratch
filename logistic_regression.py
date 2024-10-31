import math 
import numpy as np 
import pandas as pd 

def sigmoid(w, x, b):
    w = np.array(w)
    x = np.array(x)
    
    return 1 / (1 + np.exp(-(np.dot(x, w) + b)))

def logloss(real_vls, preds):
    epsilon = 1e-15
    y_pred = np.clip(preds, epsilon, 1 - epsilon)
    y_real = np.array(real_vls)
    m = len(y_real)
    loss = -np.sum(y_real * np.log(y_pred) + (1 - y_real) * np.log(1 - y_pred)) / m
    return loss

def logistic_gradient(df, target, iters=100, lr=0.001):
    nrows, ncols = df.shape
    x = df.drop(columns=target).values
    y = df[target].values
    w = np.zeros(ncols - 1)
    b = 0
    log_loss_arr = []

    for i in range(iters): 
        y_preds = sigmoid(w, x, b)
        log_loss = logloss(y, y_preds)
        log_loss_arr.append(log_loss)
        for j in range(ncols - 1):
            w[j] -= lr * np.sum((y_preds - y) * x[:, j])
        b -= lr * np.sum(y_preds - y)
    
    return w, b


def logistic_regressor(x, w, b):
    y_preds = []
    for i in range(len(x)):
        y_pred = sigmoid(w, x[i], b)
        if y_pred > 0.5: 
            y_preds.append(1)
        else:
            y_preds.append(0)
    
    return y_preds

df = pd.DataFrame({
    'height': [42, 21, 77, 65],
    'weight': [42, 11, 53, 50],
    'target': [0, 1, 1, 0] 
})

x = df.drop(columns=['target'])
y = df['target']
w, b = logistic_gradient(df, 'target')

print("Model parameters: ", w, b)
print("Predictions: ", logistic_regressor(x.values, w, b))