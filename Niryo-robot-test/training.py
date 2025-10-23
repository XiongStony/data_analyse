import torch
from tqdm.notebook import tqdm  # Import tqdm for progress bar


def full_batch_train(
        model,optimizer,
        criterion,
        num_epochs,
        X_train_tensor,y_train_tensor,
        X_r_ver_tensor,y_r_ver_tensor
    ):
    train_losses = []
    verify_losses = []

    pbar = tqdm(range(int(num_epochs)), desc="Training", leave=True)
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        # —— 2. 验证 —— #
        model.eval()
        with torch.no_grad():
            verify_outputs = model(X_r_ver_tensor)
            verify_loss = criterion(verify_outputs, y_r_ver_tensor)
            verify_losses.append(verify_loss.item())
        # —— 3. 更新 tqdm 的显示 —— #
        pbar.set_postfix(train_loss=loss.item(), verify_loss=verify_loss.item())
    return model, train_losses, verify_losses


def batch_train(model,optimizer,
        criterion,
        num_epochs,
        train_loader,
        verify_loader,
        accum_gradient=False,
        accum_num=4
    ):
    train_losses = []
    verify_losses = []
    pbar = tqdm(range(int(num_epochs)), desc="Training", leave=True)
    if accum_gradient:
        for epoch in pbar:
            model.train()
            optimizer.zero_grad()
            for step, (batch_X, batch_y) in enumerate(train_loader):
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)/accum_num
                loss.backward()
                if step % accum_num == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            train_losses.append(loss.item())
            # —— 2. 验证 —— #
            model.eval()
            with torch.no_grad():
                for batch_X, batch_y in verify_loader:
                    verify_outputs = model(batch_X)
                verify_loss = criterion(verify_outputs, batch_y)
                verify_losses.append(verify_loss.item())
            # —— 3. 更新 tqdm 的显示 —— #
            pbar.set_postfix(train_loss=loss.item(), verify_loss=verify_loss.item())
    else:
        for epoch in pbar:
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            train_losses.append(loss.item())
            # —— 2. 验证 —— #
            model.eval()
            with torch.no_grad():
                for batch_X, batch_y in verify_loader:
                    verify_outputs = model(batch_X)
                verify_loss = criterion(verify_outputs, batch_y)
                verify_losses.append(verify_loss.item())
            # —— 3. 更新 tqdm 的显示 —— #
            pbar.set_postfix(train_loss=loss.item(), verify_loss=verify_loss.item())
    return model,train_losses,verify_losses