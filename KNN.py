import pandas as pd 
import numpy as np 


def KNN(df: pd.DataFrame,k:int,target: str,test_pt: np.array) -> list: 

    m,n = df.shape 
    x = df.drop(columns=[target]).values 
    y= df[target].values 
    
    distances = np.sqrt(np.sum((x - test_pt)**2, axis=1))
    k_indices= np.argsort(distances)[:k]

    return y[k_indices].tolist()

