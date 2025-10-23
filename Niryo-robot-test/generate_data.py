import numpy as np
import random
import matplotlib.pyplot as plt
import math
import torch

def generate_data(batch:int=2000,token:int=20,channel:int=50):
    x_vec = np.linspace(0,10,channel)
    w = np.linspace(0,50,int(batch*token))
    wx = np.outer(w,x_vec)
    np.random.seed(42) 
    Scale = np.random.random((w.shape[0],1))

    X_sin = Scale*np.sin(wx)
    X_cos = Scale*np.cos(wx)
    index = [[little+k*token for little in range(token)]for k in range(batch)]
    X_sin_new = np.stack(X_sin[index], axis=0)
    X_cos_new = np.stack(X_cos[index], axis=0)
    y_sin = np.zeros(X_sin_new.shape[0])
    y_cos = np.ones(X_cos_new.shape[0])
    y_sin = np.expand_dims(y_sin, axis=1)
    y_cos = np.expand_dims(y_cos, axis=1)
    return X_sin_new, y_sin, X_cos_new, y_cos
if __name__ == "__main__":
    data1, y1, data2, y2 = generate_data()
    print(
        'data1 = ', data1.shape,
        'y1 =', y1.shape,
        'data2 =', data2.shape,
        'y2 =', y2.shape
    )
    for data in data1[700,:10,:]:
        plt.plot(data)
    plt.show()