import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from myMLlib import set_seed,  WMSE, count_parameters, get_R
from auto_training_fun import Wor, find_weight
from NeuralNetworks import RegClassifier, Traditional
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
    if args.figures:
        plt.figure(dpi=200)
        plt.plot(signalT,alpha=0.6,label=materials[1])
        plt.plot(signalV,alpha=0.6,label=materials[0])
        plt.ylim((-0.2,0.2))
        plt.xlim((-50,3000))
        plt.legend()
        plt.grid(True)

        plt.figure(figsize=(10,6),dpi=200)
        plt.plot(time_vec*1e6,signalV,alpha=0.6,label=realmater[0], linewidth=2)
        plt.plot(time_vec*1e6,signalT,alpha=0.6,label=realmater[1], linewidth=2)
        plt.ylim((-0.2,0.2))
        plt.xlim([0, 160])
        plt.xlabel("Time($\mu s$)",fontsize = 20)
        plt.ylabel("Mean Value of Voltage(V)",fontsize = 20)
        plt.title("Time Domain Response of Experiment of Silicone Materials", fontsize=20)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.legend(fontsize = 20)
        plt.grid(True)
        # plt.savefig("/home/yuqster/files/research/figures/Transformer-160.pdf", format="pdf", bbox_inches="tight")
        plt.show()
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
    if args.figures:
        s = 50
        energy = np.cumsum(S[:s])/np.sum(S[:s])*100
        plt.figure(figsize=(6,8), dpi=200)
        plt.subplot(211)
        plt.plot(range(1,51),energy,marker = 'o',linestyle='-',markersize = 3)
        plt.title("Energy", fontsize=15)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        plt.ylabel("Percentage (%)")
        plt.xlim((0,50))
        plt.grid(True)

        plt.subplot(212)
        plt.semilogy(range(1,51),S[:s],marker = 'o',linestyle='-',markersize = 3)
        plt.grid(True)

        plt.title("Singular Value", fontsize=15)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        plt.xlim((0,50))
        plt.ylabel("Value")
        # plt.savefig("/home/yuqster/files/research/figures/Transformer_energy_singlar.pdf", format="pdf",bbox_inches="tight")
        plt.tight_layout()
        plt.show()
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
    figure_path = "figures/standard"
    k = 0
    if os.path.exists(os.path.join(figure_path, f"trend{k}.pdf")):
        k += 1
    best_reg_losses = {"epoch": [], "val": []}
    best_cls_losses = {"epoch": [], "val": []}
    for win in wins:
        set_seed(seed)
        swlist = []
        swlist_r = []
        ylist = []
        y_rlist = []
        y_depth_list = []
        y_r_depth_list = []
        for channel,label, depth in zip(channelist,labellist,depthlist):
            data = sliding_window_view(channel, axis=0, window_shape=win).transpose(0,2,1)
            depth_data = depth[win-1:]
            y = np.full(data.shape[0], label, dtype=int)
            swlist.append(data)
            ylist.append(y)
            y_depth_list.append(depth_data)
        for channel,label, depth in zip(channelist_r,label_test,depthtest):
            data = sliding_window_view(channel, axis=0, window_shape=win).transpose(0,2,1)
            depth_data = depth[win-1:]
            y = np.full(data.shape[0], label, dtype=int)
            swlist_r.append(data)
            y_rlist.append(y)
            y_r_depth_list.append(depth_data)

        X = np.concatenate(swlist, axis=0)
        X_r = np.concatenate(swlist_r, axis=0)
        y_class = np.concatenate(ylist)
        y_r_class = np.concatenate(y_rlist)
        y_depth = np.concatenate(y_depth_list)
        y_r_depth = np.concatenate(y_r_depth_list)
        y = np.stack((y_class,y_depth),axis=1)
        y_r = np.stack((y_r_class,y_r_depth),axis=1)
        print(X.shape)
        print(X_r.shape)
        print(y_class.shape)
        print(y_r_class.shape)
        print(y_depth.shape)
        print(y_r_depth.shape)
        print(y.shape)
        print(y_r.shape)


        if args.figures:

            indices1 = {}
            indices2 = {}
            unique_labels = np.unique(y_class)
            length = 20000
            Xprint = np.concatenate((X[:length,-1,:],X[length:,-1,:]),axis=0)
            X_rprint = np.concatenate((X_r[:length,-1,:],X_r[length:,-1,:]),axis=0)
            print(Xprint.shape)
            yb = y_class
            y_rb = y_r_class
            plt.figure(figsize=(10,6))
            for i,label in enumerate(unique_labels):
                # 获取当前类别下的样本索引
                indices1[materials[i]] = np.where(yb.flatten() == label)
                # 计算当前类别 X 中所有样本的特征均值（按列均值）
                mean_values = Xprint[indices1[materials[i]]].mean(axis=0)
                # 绘制均值曲线
                plt.plot(range(1, mean_values.shape[0] + 1), mean_values, label=f"{realmater[int(label)]}", alpha=0.7)
            plt.xlabel("Feature Index")
            plt.ylabel("Mean Value")
            plt.title("Mean Reduced Order Values for Each Category (Normalized)")
            plt.xticks(np.append(np.arange(0,33,4),1))
            plt.xlim([1,32])
            plt.legend()
            plt.grid(True)
            # plt.savefig("/home/yuqster/files/research/figures/Transformer_reduced_order_val.pdf", format="pdf", bbox_inches="tight")
            plt.figure(figsize=(10,6))
            for i,label in enumerate(unique_labels):
                # 获取当前类别下的样本索引
                indices2[materials[i]] = np.where(y_rb.flatten() == label)
                # 计算当前类别 X 中所有样本的特征均值（按列均值）
                mean_values = X_rprint[indices2[materials[i]]].mean(axis=0) # type: ignore
                # 绘制均值曲线
                plt.plot(mean_values, label=f"{materials[i]}",alpha = 0.7)
            plt.xlabel("Feature Index")
            plt.ylabel("Mean Value")
            plt.title("Mean Values of Xr for Each Category")
            plt.legend()
            plt.grid(True)

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
        atten_dropout = 1e-4
        cls_dropout = 0.001
        reg_dropout = 0.001
        weight_decay = 0
        model = Traditional(X.shape[-1],num_classes=2,num_heads=numhead,cls_dropout=cls_dropout, attn_dropout=atten_dropout, reg_dropout=reg_dropout).to(device)
        pre_training_learning_rate = 1e-4
        classif_cri = nn.CrossEntropyLoss() #weight=class_weights.to(device)
        optimizer = optim.Adam(model.parameters(),weight_decay=weight_decay,lr=pre_training_learning_rate)

        train_depth_weight = find_weight(reg_y_train_tensor.int(), 3)
        w_reg = 0.6
        w_cls = 1-w_reg
        print(train_depth_weight.shape)
        print(X_train_tensor.shape)
        reg_train_criterion = WMSE(train_depth_weight)
        reg_ver_criterion = nn.MSELoss()




        # Pre Training Part
        train_losses = []
        verify_cls_losses = []
        verify_reg_losses = []

        num_epochs = args.epoch
        pbar = tqdm(range(int(num_epochs)), desc="Training", leave=True)

        for epoch in pbar:
            model.train()
            logits, depth = model(X_train_tensor)      # 解包更清晰
            cls_loss = classif_cri(logits, cls_y_train_tensor)              # 分类
            reg_loss = reg_train_criterion(depth, reg_y_train_tensor)
            loss = w_cls * cls_loss + w_reg * reg_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.item())
            # —— 2. 验证 —— #
            model.eval()
            with torch.no_grad():
                ver_logits, ver_depth = model(X_ver_tensor)
                ver_cls_loss = classif_cri(ver_logits, cls_y_ver_tensor)
                ver_reg_loss = reg_ver_criterion(ver_depth, reg_y_ver_tensor)
                cls_item = ver_cls_loss.item()
                verify_cls_losses.append(cls_item)
                reg_item = ver_reg_loss.item()
                verify_reg_losses.append(reg_item)
            # —— 3. 更新 tqdm 的显示 —— #
            pbar.set_postfix(
                train_loss = loss.item(),
                class_loss = cls_item,
                reg_loss = reg_item
            )
        
## Test and Draw
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
        plt.savefig(f"{figure_path}/{k}win{win}.pdf", format="pdf", bbox_inches="tight")


        para = count_parameters(model)

        text = f"\nfigure_path = {figure_path}, seed = {seed}, r = {r},  atthead = {numhead}, atten_dropout = {atten_dropout}, \n \
        win = {win}, cls_dropout = {cls_dropout}, learning_rate = {pre_training_learning_rate}, \n \
        cls wrong prediction = {cls_wrong_predicts}, Train Loss = {loss.item()}, reg best loss = {best_reg_loss},R = {R}, \n \
            reg_dropout = {cls_dropout}, reg_weight = {w_reg}, mode parameters = {para}\n\n"
        
        with open("auto_train.txt","a") as f:
            f.write(text)

        best_reg_losses["epoch"].append(reg_min_idx)
        best_reg_losses["val"].append(best_reg_loss)
        best_cls_losses["epoch"].append(cls_min_idx)
        best_cls_losses["val"].append(best_cls_loss)

        fig = plt.figure(dpi=150)
        ax = fig.add_subplot(211)
        ax.plot(range(5,win+1),best_reg_losses["val"],"--o")
        ax.set_title("Best Regression Loss")
        ax = fig.add_subplot(212)
        ax.plot(range(5,win+1),best_cls_losses["val"],"--o")
        ax.set_title("Best Classification Loss")
        plt.tight_layout()
        plt.savefig(f"{figure_path}/trend{k}.pdf", format="pdf", bbox_inches="tight")
        print("Win =", win)
