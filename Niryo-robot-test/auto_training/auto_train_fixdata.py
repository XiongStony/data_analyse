import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from myMLlib import set_seed,  WMSE, count_parameters, get_R
from auto_training_fun import Wor, get_window_Xy
from NeuralNetworks import RegClassifier, Traditional, LastToken, W2qLastToken
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix,  roc_curve, roc_auc_score
import torch
from torch import optim, nn
# from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import yaml
import os
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm  # Import tqdm for progress bar
import pandas as pd
import argparse
from copy import deepcopy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, default="No")
    parser.add_argument("--figures",type=bool, default=False)
    parser.add_argument("--epoch", type=int, default=30000)
    args = parser.parse_args()

    with open('../hyperparameters.yml', 'r') as file:
        all_parameters = yaml.safe_load(file)
        parameters = all_parameters["The_third_NeedleARM"]
    materials = parameters["materials"]
    realmater = parameters["realmater"]
    dir_path = parameters["dir_path"]
    
    with open("parameters.yml",'r') as file:
        project_parameters = yaml.safe_load(file)
        parameters = project_parameters["auto_training_fixdata"]
    project_name = parameters["project_name"]
    fixed_win = parameters["fixed_win"]
    materials = list(materials.values())
    print(materials)
    print(realmater)

    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 20,
        'axes.labelsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 14
    })

        
    name = "/home/yuqster/files/research/dataset/realtest_mwt8/loop-0_Tissue_position-1-18depth_mv.npy"
    a = np.load(name)
    print(a.shape)
    time_vec = pd.read_csv('/home/yuqster/files/research/dataset/New_bloodmeat16_right/time_vector.csv', usecols=[1]).values
    time_vec = time_vec.squeeze()
    print(time_vec)
    print(time_vec.shape)

    # Load Data
    seed = 42
    set_seed(seed)
    two_label_depth = []
    folderpath = [os.path.join(dir_path,x) for x in ("realtest_mwt9","realtest_mwt10")]
    two_data = []
    folderloops =  (100, 100)
    two_labels = []
    for loopnum,folder in zip(folderloops,folderpath):
        data = []
        label_depth = []
        label = []
        for y, mater in enumerate(materials):
            for loop in range(loopnum):
                for position in range(6):
                    mv = Wor(1)
                    channel = []
                    depth_l = []
                    for depth in range(16,21):
                        while True:
                            filename = f"loop-{loop}_{mater}_position-{position}-{depth}depth{mv.data}.npy"
                            filepath = os.path.join(folder,filename)
                            signals = np.load(filepath)
                            channel.append(signals)
                            depth_l.append(np.repeat(depth-16, repeats=signals.shape[0]))
                            mv.move()
                            if mv.i == 0: 
                                break
                    data.append(np.concatenate(channel,axis=0))
                    label.append(y)
                    label_depth.append(np.concatenate(depth_l))
        two_data.append(data)
        two_labels.append(label)
        two_label_depth.append(label_depth)



    print(len(two_data[0]))
    print(two_data[0][2].shape)

    two_shapes = []
    for dataset in two_data:
        shapes = []
        for data in dataset:
            shapes.append(data.shape[0])
        two_shapes.append(shapes)

    idx = {"train":list(range(0,400)) + list(range(600,1000)), "test":list(range(400,600)) + list(range(1000,1200))}
    mater_idx = {"V":list(range(400)) + list(range(800,1200)), "T":list(range(400, 800)) + list(range(1200,1600))}
    print(idx["train"])
    print(idx["test"])
    print(mater_idx["V"])
    print(mater_idx["T"])



    shapelist = [two_shapes[i][j] for i in range(2) for j in idx["train"]]
    shape_test = [two_shapes[i][j] for i in range(2) for j in idx["test"]]

    datalist = [two_data[i][j] for i in range(2) for j in idx["train"]]
    data_test = [two_data[i][j] for i in range(2) for j in idx["test"]]

    depthlist = [two_label_depth[i][j] for i in range(2) for j in idx["train"]]
    depthtest = [two_label_depth[i][j] for i in range(2) for j in idx["test"]]

    labellist =  [two_labels[i][j] for i in range(2) for j in idx["train"]]
    label_test = [two_labels[i][j] for i in range(2) for j in idx["test"]]

    print(len(shapelist))
    print(len(shape_test))

    print(len(labellist))
    print(len(label_test))

    print(len(depthlist))
    print(len(depthtest))

    print(depthlist[0].shape)
    print(datalist[0].shape)
    signalV = np.concatenate([datalist[i] for i in mater_idx["V"]],axis=0).mean(axis=0)
    signalT = np.concatenate([datalist[i] for i in mater_idx["T"]],axis=0).mean(axis=0)

    snapshot = np.concatenate(datalist,axis=0).squeeze()[:,:2500]
    snapshot_r = np.concatenate(data_test,axis=0).squeeze()[:,:2500]
    print(snapshot.shape)
    print(snapshot_r.shape)
    print(len(depthlist))
    print(len(depthtest))
    print(snapshot.shape[0] + snapshot_r.shape[0])
    # Save

    # Load SVD modes
    recalculate = False
    savefolder = "../mwtfiles"
    savepathV = os.path.join(savefolder,"Third_mwt_V.npy")
    savepathS = os.path.join(savefolder,"Third_mwt_S.npy")
    if not os.path.exists(savepathV) or recalculate:
        _, S, Vt = np.linalg.svd(snapshot, full_matrices=False)
        V = Vt.T
        np.save(savepathS, S)
        np.save(savepathV, V)
    else:
        S = np.load(savepathS)
        V = np.load(savepathV)

    r = 32
    Vr = V[:,:r]
    print(Vr.shape)
    Xpp = snapshot @ Vr
    X_rpp = snapshot_r @ Vr
    print(Xpp.shape)
    print(X_rpp.shape)
    scaler = StandardScaler()

    Xp = scaler.fit_transform(Xpp)
    X_rp = scaler.transform(X_rpp)
    def indices(shapelist):
        i = 0
        for shape in shapelist:
            yield list(range(i,i+shape))
            i = i+shape
    channelist = []
    channelist_r = []
    for ind in indices(shapelist):
        data = Xp[ind]
        channelist.append(data)
    for ind in indices(shape_test):
        data = X_rp[ind]
        channelist_r.append(data)
    print(len(channelist))
    print(len(channelist_r))
    print(channelist[0].shape)


    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) 
        else "cpu"
    )
    print("Using device:", device)
    
    wins = range(5,16)

# Begin Training
    best_train_losses = []
    best_reg_losses = {"epoch": [], "val": []}
    best_cls_losses = {"epoch": [], "val": []}
    X_pre, y_pre = get_window_Xy(fixed_win, channelist, labellist, depthlist)
    X_r_pre, y_r_pre = get_window_Xy(fixed_win, channelist_r,label_test,depthtest)
    for i, win in enumerate(wins):
        X = X_pre[:,-win:,:]
        y = y_pre
        X_r, y_r = X_r_pre[:,-win:,:], y_r_pre

        X_ver,X_te,y_ver,y_te = train_test_split(X_r, y_r, test_size=0.5,shuffle=True)
        X_train_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        cls_y_train_tensor = torch.tensor(y[:,0], dtype=torch.long).squeeze().to(device)
        reg_y_train_tensor = torch.tensor(y[:,1], dtype=torch.float32).squeeze().to(device)

        X_ver_tensor = torch.tensor(X_ver, dtype=torch.float32).to(device)
        cls_y_ver_tensor = torch.tensor(y_ver[:,0], dtype=torch.long).squeeze().to(device)
        reg_y_ver_tensor = torch.tensor(y_ver[:,1], dtype=torch.float32).squeeze().to(device)

        X_te_tensor = torch.tensor(X_te, dtype=torch.float32).to(device)
        cls_y_te_tensor = torch.tensor(y_te[:,0], dtype=torch.long).squeeze().to(device)
        reg_y_te_tensor = torch.tensor(y_te[:,1], dtype=torch.float32).squeeze().to(device)


        idx = torch.randperm(X_train_tensor.size(0))  # 随机打乱索引
        X_train_tensor = X_train_tensor[idx]
        cls_y_train_tensor = cls_y_train_tensor[idx]
        reg_y_train_tensor = reg_y_train_tensor[idx]
        print(X_train_tensor.shape)
        print(cls_y_train_tensor.shape)
        print(X_te_tensor.shape)
        print(reg_y_te_tensor.shape)


        numhead = 2
        atten_dropout = 0.2
        cls_dropout = 0.1
        reg_dropout = 0.2
        weight_decay = 0
        model = Traditional(X.shape[-1],num_classes=2,num_heads=numhead,cls_dropout=cls_dropout, attn_dropout=atten_dropout, reg_dropout=reg_dropout).to(device)
        pre_training_learning_rate = 1e-4
        classif_cri = nn.CrossEntropyLoss() #weight=class_weights.to(device)
        optimizer = optim.Adam(model.parameters(),weight_decay=weight_decay,lr=pre_training_learning_rate)

        w_reg = 0.8
        w_cls = 1-w_reg
        print(X_train_tensor.shape)
        reg_criterion = nn.MSELoss()


        # Pre Training Part
        train_losses = []
        verify_cls_losses = []
        verify_reg_losses = []
        best_reg_loss = float('inf')

        num_epochs = args.epoch
        pbar = tqdm(range(int(num_epochs)), desc="Training", leave=True)

        for epoch in pbar:
            model.train()
            logits, depth = model(X_train_tensor)      # 解包更清晰
            cls_loss = classif_cri(logits, cls_y_train_tensor)              # 分类
            reg_loss = reg_criterion(depth, reg_y_train_tensor)
            loss = w_cls * cls_loss + w_reg * reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            # —— 2. 验证 —— #
            model.eval()
            with torch.no_grad():
                ver_logits, ver_depth = model(X_ver_tensor)
                ver_cls_loss = classif_cri(ver_logits, cls_y_ver_tensor)
                ver_reg_loss = reg_criterion(ver_depth, reg_y_ver_tensor)
                cls_item = ver_cls_loss.item()
                verify_cls_losses.append(cls_item)
                reg_item = ver_reg_loss.item()
                verify_reg_losses.append(reg_item)
            if reg_item < best_reg_loss and epoch > 5000:
                best_reg_loss = reg_item
                best_state = deepcopy(model.state_dict())
            # —— 3. 更新 tqdm 的显示 —— #
            pbar.set_postfix(
                train_loss = loss.item(),
                class_loss = cls_item,
                reg_loss = reg_item
            )
        model.load_state_dict(best_state)

## Test and Draw
        if i == 0:
            figure_path = f"figures/{project_name}"
            k = 0
            while os.path.exists(os.path.join(figure_path,f"{model.__class__.__name__}_{k}_auto")):
                k += 1
            save_folder = os.path.join(figure_path,f"{model.__class__.__name__}_{k}_auto")
            model_path = os.path.join(save_folder,"models")
            os.makedirs(model_path,exist_ok=True)

        torch.save(best_state,os.path.join(model_path,f"{type(model).__name__}win_{win}.pt"))
        model.eval()
        with torch.no_grad():
            test_logits, test_depth = model(X_te_tensor)
            cls_predict_np = torch.argmax(test_logits, dim=1).cpu().numpy()
            reg_predict_np = test_depth.cpu().numpy()
            cls_y_te_np = cls_y_te_tensor.cpu().numpy()
            reg_y_te_np = reg_y_te_tensor.cpu().numpy()


        R = get_R(reg_y_te_np, reg_predict_np)

        cls_wrong_predicts = np.sum(cls_predict_np != cls_y_te_np)
        
        cls_min_idx = np.argmin(verify_cls_losses).astype(np.int32)
        best_cls_loss = verify_cls_losses[cls_min_idx]
        reg_min_idx = np.argmin(verify_reg_losses).astype(np.int32)
        best_reg_loss = verify_reg_losses[reg_min_idx]
        best_train_losses.append(loss.item())

        fig = plt.figure(dpi=150)
        ax = fig.add_subplot(111)
        ax.plot(train_losses,label="train loss")
        ax.plot(verify_cls_losses,label="cls loss")
        ax.plot(verify_reg_losses,label="reg loss")
        ax.set_xlim([0,len(train_losses)])
        ax.axvline(x=cls_min_idx, linestyle='--', color='gray',
                            label=f"Best Verify cls @ epoch {cls_min_idx} at {best_cls_loss:.2f}")
        ax.axvline(x=reg_min_idx, linestyle='--', color='pink',
                            label=f"Best Verify reg @ epoch {reg_min_idx} at {best_reg_loss:.2f}")
        ax.legend()
        plt.savefig(f"{save_folder}/win{win}.pdf", format="pdf", bbox_inches="tight")

        para = count_parameters(model)

        text = f" \n {model.__class__.__name__}, att{k} , seed = {seed}, r = {r},  atthead = {numhead}, atten_dropout = {atten_dropout}, \n \
        win = {win}, cls_dropout = {cls_dropout}, learning_rate = {pre_training_learning_rate}, \n \
        cls wrong prediction = {cls_wrong_predicts}, Train Loss = {loss.item()}, reg best loss = {best_reg_loss},R = {R}, \n \
            reg_dropout = {cls_dropout}, reg_weight = {w_reg}, mode parameters = {para}\n\n"
        
        with open(f"{save_folder}/record.txt","a") as f:
            f.write(text)


        out_dep_np, reg_np = reg_predict_np + 2, reg_y_te_np + 2
        labels = np.unique(reg_np)
        mask   = labels[:, None] == reg_np[None, :]   # (K, N)
        plt.figure(figsize=(6, 20), dpi =150)
        for i, (lab, ma) in enumerate(zip(labels, mask)):

            # 该真实标签下的预测深度
            plt.subplot(5,1,i+1)
            vals = out_dep_np[ma]

            mse = ((lab - vals)**2).mean()
            print(mse)
            plt.hist(vals, bins=50, color='blue', alpha=0.6, density=True)
            ax = plt.gca()
            ax.text(0.72, 0.78, f"MSE = {mse:.2f}",
                transform=ax.transAxes, fontsize=11,
                va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='none'))
            # 画一条真实深度的竖直线
            plt.axvline(lab, color='red', linestyle='--', linewidth=1.5,
                        label=f"True depth = {lab}")
            
            plt.axvline(vals.mean(), color ='black', linestyle='--', linewidth=1.5,
                        label=f"Expectation = {vals.mean():.2f}")

            # 让真实标签大致居中（可按需要调 delta）
            delta = 5  # 比如左右各 2 单位
            plt.xlim(lab - delta, lab + delta)
            plt.ylim(0,2.2)
            plt.xticks(range(int(lab) - delta, int(lab) + delta +1))
            plt.xlabel("Predicted depth")
            plt.ylabel("Probability Density")
            plt.grid(True, axis='y')
            plt.title(f"Probability of Predicted depth distribution", fontsize = 15)
            plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{save_folder}/{model.__class__.__name__}_att_win{win}_pdfs.pdf", format="pdf", bbox_inches="tight")


        best_reg_losses["epoch"].append(reg_min_idx)
        best_reg_losses["val"].append(best_reg_loss)
        best_cls_losses["epoch"].append(cls_min_idx)
        best_cls_losses["val"].append(best_cls_loss)

        fig = plt.figure(dpi=200,figsize=(5,10))
        ax = fig.add_subplot(311)
        ax.plot(range(5,win+1),best_train_losses,"--o")
        ax.grid(True)
        ax.set_title("Best Training Loss")
        ax = fig.add_subplot(312)
        ax.plot(range(5,win+1),best_reg_losses["val"],"--o")
        ax.grid(True)
        ax.set_title("Best Regression Loss")
        ax = fig.add_subplot(313)
        ax.plot(range(5,win+1),best_cls_losses["val"],"--o")
        ax.grid(True)
        ax.set_title("Best Classification Loss")
        plt.tight_layout()
        plt.savefig(f"{save_folder}/trend.pdf", format="pdf", bbox_inches="tight")
        print("Win =", win)
