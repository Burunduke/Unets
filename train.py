import time
from glob import glob
import gc
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.backends.cudnn import benchmark

from UNET.data import DriveDataset
from UNET.model import UNet, R2UNet, AttUNet, R2AttUNet, UNet2, ResUnet, ResUnetPlusPlus, NestedUNet
from UNET.loss import DiceBCELoss
from UNET.utils import seeding, create_dir, epoch_time
import matplotlib.pyplot as plt


def train(model, name, num_epochs, train_loader, valid_loader, optimizer, loss_fn, device, checkpoint):

    with tqdm(total=(len(train_loader)+len(valid_loader))*num_epochs, position=0, leave=True) as pbar:
        for epoch in range(checkpoint['epochs'][-1] + 1, checkpoint['epochs'][-1] + 1 + num_epochs):

            # train loss
            model.train()
            train_loss = 0.0
            start_time = time.time()

            for batch_num, (x, y) in enumerate(train_loader):
                x = x.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.float32)
                optimizer.zero_grad()
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()
                train_loss = (train_loss*batch_num + loss.item()) / (batch_num + 1)
                pbar.set_description("Epoch: %d, training batch num: %2d, Loss: %f" % (epoch, batch_num, train_loss))
                pbar.update()
            checkpoint["train_losses"].append(train_loss)

            # valid loss
            valid_loss = 0.0
            model.eval()
            with torch.no_grad():
                for batch_num, (x, y) in enumerate(valid_loader):
                    x = x.to(device, dtype=torch.float32)
                    y = y.to(device, dtype=torch.float32)

                    y_pred = model(x)
                    loss = loss_fn(y_pred, y)
                    valid_loss = (valid_loss * batch_num + loss.item()) / (batch_num + 1)
                    pbar.set_description("Epoch: %d, valid batch num: %2d, Loss: %f" % (epoch, batch_num, valid_loss))
                    pbar.update()
            checkpoint["valid_losses"].append(valid_loss)

            """saving"""
            if checkpoint["valid_losses"][-1] < checkpoint['best_valid_loss']:
                data_str = f"\n{name} valid loss improved from {checkpoint['best_valid_loss']:2.4f} to {checkpoint['valid_losses'][-1]:2.4f}"
                print(data_str)

                checkpoint['best_valid_loss'] = checkpoint["valid_losses"][-1]
                checkpoint['model_state_dict'] = model.state_dict()
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
                torch.save(checkpoint, check)
                print(f"Saved to {check}")

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            now_time = time.strftime("%X", time.localtime(time.time()))
            data_str = f"\n{now_time} : Epoch: {epoch:02} | Epoch time: {epoch_mins}m {epoch_secs}s"
            data_str += f'\tTrain loss: {checkpoint["train_losses"][-1]:.3f}'
            data_str += f"\tValid loss: {checkpoint['valid_losses'][-1]:.3f}\n"
            print(data_str)
            gc.collect()
            torch.cuda.empty_cache()
            checkpoint['epochs'].append(epoch)
        pbar.close()


if __name__ == "__main__":
    """random and data folder"""
    seeding(42)
    create_dir("UNET\\checkpoints")

    """hyperparameters"""
    H = 256
    W = 256
    size = (H, W)
    batch_size = 1
    start_train_size = None
    end_train_size = None
    start_valid_size = start_train_size
    end_valid_size = None #end_train_size*3//10
    num_epochs = 20
    lr = 5e-5
    checkpoint_path = {
        "unet": "UNET/checkpoints/checkpoint_unet",
        "unet2": "UNET/checkpoints/checkpoint_unet2",
        "r2unet": "UNET/checkpoints/checkpoint_r2unet",
        "atunet": "UNET/checkpoints/checkpoint_atunet",
        "resunet": "UNET/checkpoints/checkpoint_resunet",
        "resunetplusplus": "UNET/checkpoints/checkpoint_resunetplusplus",
        "nestedunet": "UNET/checkpoints/checkpoint_nestedunet",
        "r2atunet": "UNET/checkpoints/checkpoint_r2atunet"}
    load = True


    """load dataset"""
    augs = ["original", 'multidim'] #, "clahe", "grey", 'r', 'g', 'b', 'sobel', "roberts", 'prewitt', 'sato', 'meijering', 'scharr', 'unsharp_mask'
    for aug in augs:


        a = glob(r"D:\Study\Python\unets\new_data\clahe\train\image\*")
        train_x = sorted(glob(f"D:\\Study\\Python\\unets\\new_data\\{aug}\\train\\image\\*"))[start_train_size:end_train_size]
        train_y = sorted(glob(f"D:\\Study\\Python\\unets\\new_data\\{aug}\\train\\mask\\*"))[start_train_size:end_train_size]

        valid_x = sorted(glob(f"D:\\Study\\Python\\unets\\new_data\\{aug}\\valid\\image\\*"))[start_valid_size:end_valid_size]
        valid_y = sorted(glob(f"D:\\Study\\Python\\unets\\new_data\\{aug}\\valid\\mask\\*"))[start_valid_size:end_valid_size]

        data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}"
        print(data_str)

        """"dataset and loader"""
        train_dataset = DriveDataset(train_x, train_y, mode=1)
        valid_dataset = DriveDataset(valid_x, valid_y, mode=1)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        """model"""
        device = torch.device('cuda')
        models = {
            "unet": UNet(img_ch=3), #y
            #"r2unet": R2UNet(), #n
            "atunet": AttUNet(), #y
            "unet2": UNet2(), #y
            #"resunet": ResUnet(), #y
            #"resunetplusplus": ResUnetPlusPlus(), #y
            "nestedunet": NestedUNet(), #y
            #"r2atunet" : R2AttUNet() #n
            }

        for name, model in models.items():
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            check = checkpoint_path[name] + f"_{aug}.tar"
            print(f"Training {name} on {aug} data\n")

            """loading if available"""
            if load:
                checkpoint = torch.load(check)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                torch.backends.cudnn.benchmark = True
            else:
                checkpoint = {
                    'epochs': [0],
                    'train_losses': [1],
                    'valid_losses': [1],
                    'model_state_dict': 0,
                    'optimizer_state_dict': 0,
                    'best_valid_loss': float("inf"),
                }
            start_epoch = checkpoint['epochs'][-1]+1
            torch.cuda.empty_cache()

            sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, verbose=True)
            loss_fn = DiceBCELoss()


            """training"""
            train(model, name, num_epochs, train_loader, valid_loader, optimizer, loss_fn, device, checkpoint)

    # plt.plot(checkpoint['epochs'], checkpoint['train_losses'], label="train loss")
    # plt.plot(checkpoint['epochs'], checkpoint['valid_losses'], label="valid loss")
    # plt.xlabel("Эпохи")
    # plt.ylabel("Ошибка")
    # plt.legend(loc='best')
    # plt.show()
