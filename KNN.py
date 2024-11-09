import pandas as pd 
import numpy as np 

def cosine_similarity(x,pt):
    return np.dot(x, pt) / (np.linalg.norm(x) * np.linalg.norm(pt))

def euclidean_distance(x,pt):
    return np.sqrt(np.sum(x-pt)**2,axis= 1)

def manhattanDistance(x,pt):
    return np.abs(np.sum(x-pt))

def KNN(df: pd.DataFrame,k:int,target: str,test_pt: np.array) -> list: 

    m,n = df.shape 
    x = df.drop(columns=[target]).values 
    y= df[target].values 
    
    distances = euclidean_distance(x,test_pt)
    k_indices= np.argsort(distances)[:k]

    return y[k_indices].tolist()

