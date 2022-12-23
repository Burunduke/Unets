import os
import time
from glob import glob

import torch
from torch.utils.data import DataLoader
from torch.backends.cudnn import benchmark
import torch.nn as nn

from UNET.data import DriveDataset
from UNET.model import UNet
from UNET.loss import DiceLoss, DiceBCELoss
from UNET.utils import seeding, create_dir, epoch_time
import matplotlib.pyplot as plt


def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0

    model.train()
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(loader)
    return epoch_loss


def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(loader)

    return epoch_loss


if __name__ == "__main__":
    """random and data folder"""
    seeding(42)
    create_dir("UNET\\files")

    """hyperparameters"""
    H = 512
    W = 512
    size = (H, W)
    batch_size = 1
    start_train_size = 128
    end_train_size = 512
    num_epochs = 10
    lr = 1e-4
    checkpoint_path = "UNET/files/checkpoint.tar"
    load = True

    """load dataset"""
    a = glob("D:\\Study\\Python\\unet\\new_data\\train\\image\\*")
    train_x = sorted(glob("D:\\Study\\Python\\unet\\new_data\\train\\image\\*"))[start_train_size:end_train_size]
    train_y = sorted(glob("D:\\Study\\Python\\unet\\new_data\\train\\mask\\*"))[start_train_size:end_train_size]

    valid_x = sorted(glob("D:\\Study\\Python\\unet\\new_data\\valid\\image\\*"))
    valid_y = sorted(glob("D:\\Study\\Python\\unet\\new_data\\valid\\mask\\*"))

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

    """"dataset and loader"""
    train_dataset = DriveDataset(train_x, train_y)
    valid_dataset = DriveDataset(valid_x, valid_y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    best_valid_loss = float("inf")

    """model"""
    device = torch.device('cuda')
    model = UNet()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    """loading if available"""
    if load:
        checkpoint = torch.load(checkpoint_path)
        epochs = checkpoint['epochs']
        start_epoch = epochs[-1]
        train_loss = checkpoint['train_losses']
        valid_loss = checkpoint['valid_losses']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_valid_loss = checkpoint['best_valid_loss']
        model.train()
        torch.backends.cudnn.benchmark = True
    else:
        start_epoch = 1
        train_loss = []
        valid_loss = []
        epochs = []
    torch.cuda.empty_cache()

    sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, verbose=True)
    loss_fn = DiceBCELoss()


    """training"""
    for epoch in range(start_epoch, start_epoch + num_epochs):
        start_time = time.time()

        train_loss.append(train(model, train_loader, optimizer, loss_fn, device))
        valid_loss.append(evaluate(model, valid_loader, loss_fn, device))
        epochs.append(epoch)

        """saving"""
        if valid_loss[-1] < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss[-1]}"
            print(data_str)

            best_valid_loss = valid_loss[-1]
            torch.save({
                'epochs': epochs,
                'train_losses': train_loss,
                'valid_losses': valid_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_valid_loss': best_valid_loss
            }, checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        now_time = time.strftime("%X", time.localtime(time.time()))
        data_str = f"{now_time} : Epoch: {epoch:02} | Epoch time: {epoch_mins}m {epoch_secs}s"
        data_str += f'\tTrain loss: {train_loss[-1]:.3f}\n'
        data_str += f"\tValid loss: {valid_loss[-1]:.3f}\n"
        print(data_str)

    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, valid_loss, label="valid loss")
    plt.xlabel("Эпохи")
    plt.ylabel("Ошибка")
    plt.legend(loc='best')
    plt.show()
