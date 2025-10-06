import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from pymor.algorithms.ei import deim
from pymor.vectorarrays.numpy import NumpyVectorSpace
from torch.utils.data import Dataset, Sampler
import os
import pandas as pd
def set_seed(seed=42,deterministic = True, benchmark = False):
    torch.manual_seed(seed)             # CPU随机性
    torch.cuda.manual_seed(seed)        # GPU随机性（单卡）
    torch.cuda.manual_seed_all(seed)    # GPU随机性（多卡）
    np.random.seed(seed)                # NumPy随机性
    random.seed(seed)                   # Python随机性
    torch.backends.cudnn.deterministic = deterministic  # 固定卷积算法
    torch.backends.cudnn.benchmark = benchmark     # 禁止自动寻找最优算法（为了可重复性）
def plt_confusion(cm, objects, fs=(8,6)):
    plt.figure(figsize=fs)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=objects,
                yticklabels=objects)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
def DEIM(Ur):
    n, r = Ur.shape            # Ur: (n, r)，列为基向量
    U = NumpyVectorSpace(n).from_numpy(Ur.T)  # 传入形状 (r, n)
    dofs, _, _ = deim(U, modes=r, pod=False)
    return dofs                # 若需要选择矩阵：np.eye(n)[:, dofs]

def plt_loss(train_losses, test_losses=None, verifying_losses=None):
    plt.figure(figsize=(10, 5))
    if test_losses is not None:
        plt.plot(test_losses, label="Testing Loss")
    if verifying_losses is not None:
        plt.plot(verifying_losses, label="Verifying Loss")
        
        # 找到 verifying_losses 最低点对应的索引
        best_epoch = int(np.argmin(verifying_losses))
        # 在该索引处画一条竖直虚线
        plt.axvline(x=best_epoch, linestyle='--', color='gray',
                    label=f"Best Verify @ epoch {best_epoch} at {verifying_losses[best_epoch]:.2f}")
        
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()
    plt.show()

class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=False)
        # 分类头：把最后一个时间步的隐藏状态映射到类别数
        self.classifier = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        output, (hn, cn) = self.lstm(x)
        # hn: [num_layers, batch, hidden_size]
        last_hidden = hn[-1]                # 取最后一层的隐藏状态 [batch, hidden_size]
        logits = self.classifier(last_hidden)  # [batch, num_classes]
        return logits
    

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    


class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size
        # 假设二分类，标签为 0 / 1
        self.pos_idx = [i for i,l in enumerate(labels) if l==1]
        self.neg_idx = [i for i,l in enumerate(labels) if l==0]
        assert batch_size % 2 == 0
        self.half = batch_size // 2

    def __iter__(self):
        random.shuffle(self.pos_idx)
        random.shuffle(self.neg_idx)
        # 根据较多类别的长度决定迭代次数
        iters = max(len(self.pos_idx), len(self.neg_idx)) // self.half + 1
        for i in range(iters):
            batch = []
            # 从各自列表循环取 half 个
            for j in range(self.half):
                batch.append(self.pos_idx[(i*self.half + j) % len(self.pos_idx)])
                batch.append(self.neg_idx[(i*self.half + j) % len(self.neg_idx)])
            yield batch

def load_data_list(folderpath, materials, N=None):
    # 如果调用时没有传入 N，就根据 folderpath 的长度生成一个全是 3e5 的列表
    if N is None:
        N = [3e5] * len(folderpath)
    N = [int(x) for x in N]
    data_list = []
    labels = []
    for j, folder in enumerate(folderpath):
        data_list.append([])   # 动态创建子列表
        labels.append([])      # 同上
        files = os.listdir(folder)
        if len(files) > N[j]:
            files = files[0:N[j]]
        for i in range(len(materials)):
            matching_files = [f for f in files if materials[i] in f]
            for path in matching_files:
                data = pd.read_csv(f'{folder}/{path}')
                trans = data.values
                data_list[j].append(trans)
                labels[j].append(i)
    return data_list, labels
def load_to_alist(folderpath,materials, N = 3e5):
    N = int(N)
    data_list = []
    labels = []
    files = os.listdir(folderpath)
    if len(files) > N:
        files = files[0:N]
    for i in range(len(materials)):
        matching_files = [f for f in files if materials[i] in f]
        for path in matching_files:
            data = pd.read_csv(f'{folderpath}/{path}')
            trans = data.values
            data_list.append(trans)
            labels.append(i)
    return data_list, labels
def rebalance_weight(y):
    labels = torch.tensor(y,dtype=torch.long).squeeze()  # torch.Tensor, dtype=torch.long
    labels_range = len(np.unique(y))
    class_counts = torch.bincount(labels, minlength=labels_range).float()
    # 2) 计算权重：常见做法是取 inverse frequency
    class_weights = 1.0 / class_counts                              
    # （可选）归一化，让总和为 C
    class_weights = class_weights / class_weights.sum() * labels_range
    return class_weights, labels_range

def plotCWTscalogram(time,freqs,cwtmatr, frerange = None,figuresize = (10,6), Name ='CWT Scalogram',contourplot = False, cmapnum = 15):
    plt.figure(figsize=figuresize)
    if frerange != None:
        inbeg = np.searchsorted(freqs, frerange[0])
        inend = np.searchsorted(freqs, frerange[1])
        target = np.abs(cwtmatr[inbeg:inend,:])
        fretarget = freqs[inbeg:inend]*1e-3
        plt.ylim([frerange[0]*1e-3,frerange[1]*1e-3])
    else:
        target = np.abs(cwtmatr)
        fretarget = freqs*1e-3
    if contourplot == True:
        CS = plt.contourf(time, fretarget, target, levels=cmapnum, cmap='viridis')
    else:
        CS = plt.pcolormesh(time, fretarget, target, shading='auto')
    plt.colorbar(CS)
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [kHz]')
    plt.title(Name)
    plt.show()
def layer_activation(model, device, data, fclayer='fc1'):
    activations = {}

    # 工厂函数，给定名字 name，返回一个 forward hook
    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu().numpy()
        return hook

    # 动态获取 model 上名为 fclayer 的子模块
    layer = getattr(model, fclayer)
    handle = layer.register_forward_hook(get_activation(fclayer))

    # 前向传播一批数据
    _ = model(torch.tensor(data, dtype=torch.float32).to(device))

    handle.remove()
    output = activations[fclayer]
    return output
def neuron_activation(model, device, data, fclayer='fc1', neuron_index=0):
    activations = {}

    # 注册 forward hook，记录该层的输出
    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu().numpy()
        return hook

    layer = getattr(model, fclayer)
    handle = layer.register_forward_hook(get_activation(fclayer))

    # 前向传播
    _ = model(torch.tensor(data, dtype=torch.float32).to(device))

    # 获取所有样本在该神经元上的激活值
    layer_output = activations[fclayer]   # shape: (batch_size, num_neurons)
    
    if neuron_index >= layer_output.shape[1]:
        raise ValueError(f"neuron_index {neuron_index} 超出该层维度 {layer_output.shape[1]}")
    
    neuron_activations = layer_output[:, neuron_index]  # shape: (batch_size,)
    handle.remove()
    return neuron_activations
    