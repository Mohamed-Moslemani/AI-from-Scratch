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

##for multiple features
def MultipleGradientDescent(df,target,lr,iters):
    m,n = df.shape
    w = [0]*n
    x = df.drop(columns=[target])
    y = df[target]   
    b = 0 
    for i in range(iters): 
        for j in range(m):
            for k in range(x.shape[1]):
                w[k] = w[k] + (lr/m)*(y.iloc[j] - w[k]*x.iloc[j, k] - b) * x.iloc[j, k] ##update equation of w
            b = b + (lr / m) * (y.iloc[j] - sum(w[k] * x.iloc[j, k] for k in range(x.shape[1])) - b) ##update equation of b 
    
    print(f"weights: {w}")
    print(f'bias: {b}')

    return w,b,x,y


def linear_regressor(w,b,x,y):
    y_preds = []
    for i in range(len(y)):
        prediction = sum(w[j] * x.iloc[i, j] for j in range(x.shape[1])) + b
        y_preds.append(prediction)
    return y_preds

data = {
    'name': [3, 433, 53],
    'age': [24, 25, 23],
    'score': [85, 88, 90]
}
df = pd.DataFrame(data)

w,b,x,y = MultipleGradientDescent(df,target='age',lr=0.0001,iters=50)
print("_____________________________________")
print("Predicted values: ")
print(list(linear_regressor(w,b,x,y)))

