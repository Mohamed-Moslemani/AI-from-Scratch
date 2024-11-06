import pandas as pd 
import numpy as np 

def linearregressionGD(df: pd.DataFrame, iters: int, target: str, learning_rate: float):
    m,n = df.shape
    x = df.drop(columns=[target]).values 
    y = df[target].values
    w = [0]*(n-1)
    b = 0 

    for i in range(iters):
        y_pred = np.dot(x,w) + b 
        dw = np.zeros(n-1)
        db= 0 
        for j in range(n-1):
            dw[j] = (1/m)*sum((y_pred - y)*x[:,j])
        db = (1/m)*sum(y_pred-y)

        w = w - learning_rate*dw 
        b = b - learning_rate*db 
    
    return w,b 

def linearRegressor(w,b,x):
    return np.dot(x,w) + b


df = {
    'age': [13,15,28,43],
    'height': [173,155,180,193], 
    'weight': [55,67,70,85], 
    'handwidth': [0.5,0.33,0.78,1.23]
}

df = pd.DataFrame(df)

w,b = linearregressionGD(df, 100, 'handwidth',0.001)

print(f"optimized weight: {w}")
print("-"*50)
print(f"optimized bias: {b}")

test_data = {
    'age': [18, 22, 30, 40, 45],
    'height': [165, 170, 185, 195, 200],
    'weight': [58, 65, 72, 80, 88]
}
test_df = pd.DataFrame(test_data)
x_test = test_df.values 
print("-"*50)
print(f'predictions: {linearRegressor(w,b,x_test)}')