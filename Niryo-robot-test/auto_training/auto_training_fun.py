import torch
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