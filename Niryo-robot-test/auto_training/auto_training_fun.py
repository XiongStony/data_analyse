import torch
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
def get_window_Xy(win, channelist, labellist, depthlist):
    swlist = []
    ylist = []
    y_depth_list = []
    for channel, label, depth in zip(channelist,labellist,depthlist):
        data = sliding_window_view(channel, axis=0, window_shape=win).transpose(0,2,1)
        depth_data = depth[win-1:]
        yy = np.full(data.shape[0], label, dtype=int)
        swlist.append(data)
        ylist.append(yy)
        y_depth_list.append(depth_data)
    X = np.concatenate(swlist, axis=0)
    y_class = np.concatenate(ylist)
    y_depth = np.concatenate(y_depth_list)
    y = np.stack((y_class,y_depth),axis=1)
    print(X.shape)
    print(y_class.shape)
    print(y_depth.shape)
    print(y.shape)
    return X, y


class Wor():
    def __init__(self,i = None):
        self.a = {0:"_mv",1:""}
        self.i = 0 if i is None else i
        self.data = self.a[self.i]
    def move(self):
        self.i+=1
        if self.i >= 2:
            self.i = 0
        self.data = self.a[self.i]
        return self
def find_weight(labels:torch.Tensor, x:float = 0):
    labels = labels-labels.min()
    print(labels)
    count = torch.bincount(labels)
    factor = 1/count
    factor = factor/factor.mean()
    factor = torch.sqrt(factor + x)
    factor = factor/factor.mean()
    print(factor)
    weight = factor[labels]
    weight = weight/torch.sum(weight)
    return weight
