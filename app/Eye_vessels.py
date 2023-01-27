import os
import cv2
import numpy as np
import torch
import random
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import Tk, filedialog, CENTER

from model import UNet


def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def application():
    root = Tk()
    root.title("Eye vessel mask finder")
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    # setting tkinter window size
    root.geometry("%dx%d" % (width, height))
    label2 = tk.Label(root, text='', font=("Arial", 25))
    label2.place(relx=.5, rely=.9, anchor=CENTER)

    def clicked():
        filename = filedialog.askopenfilename()
        test = image_process(filename)
        blue, green, red = cv2.split(test)
        img = cv2.merge((red, green, blue))
        im = Image.fromarray(img.astype(np.uint8))
        im_res = im.resize((width // 2, height // 2))
        imgtk = ImageTk.PhotoImage(image=im_res)
        label1 = tk.Label(root, image=imgtk)
        label1.image = imgtk
        label1.place(relx=.5, rely=.6, anchor=CENTER)
        label2.configure(text="Оригинальное изображение(Слева)\tКапилляры глаза(Справа)\nИзображение сохранено")
        label2.place(relx=.5, rely=.9, anchor=CENTER)
        btn.place(relx=.5, rely=.2, anchor=CENTER)

    btn = tk.Button(root, text="Выберите изображение!", font=("Arial", 25), command=clicked)
    btn.place(relx=.5, rely=.7, anchor=CENTER)
    root.mainloop()


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)  # (256, 256, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  # (256, 256, 3)
    return mask


def image_process(filename):
    """seeding"""
    seeding(42)

    """Hyperparams"""
    H = 256
    W = 256
    size = (W, H)
    checkpoint_path = "checkpoint_unet_original.tar"

    """load the checkpoint"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
    model.eval()

    name = filename.split("\\")[-1].split('.')[0]

    """read image"""
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    orig_size = image.shape[:2]
    image = cv2.resize(image, size)  # (256,256,3)
    cv2.imwrite(f"{name}_256.png", image)
    x = np.transpose(image, (2, 0, 1))  # (3, 256, 256)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)  # (1, 3, 256, 256)
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    x = x.to(device)

    with torch.no_grad():
        """predicting"""
        pred_y = model(x)
        pred_y = torch.sigmoid(pred_y)
        pred_y = pred_y[0].cpu().numpy()  # (1, 256, 256)
        pred_y = np.squeeze(pred_y, axis=0)  # (256, 256)
        pred_y = pred_y > 0.5
        pred_y = np.array(pred_y, dtype=np.uint8)

        """saving masks"""
        pred_y = mask_parse(pred_y)
        line = np.ones((size[1], 10, 3)) * 128

        cat_image = np.concatenate([image, line, pred_y * 255], axis=1)
        cv2.imwrite(f"{name}_all_256.png", cat_image)
        line = np.ones((orig_size[1], 10, 3)) * 128
        cat_image = np.concatenate([cv2.resize(image, orig_size), line, cv2.resize(pred_y * 255, orig_size)], axis=1)
        cv2.imwrite(f"{name}_all.png", cat_image)
        cv2.imwrite(f"{name}_mask_256.png", pred_y * 255)
    return cat_image


if __name__ == "__main__":
    application()
