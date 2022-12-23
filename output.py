import time
from glob import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm
from torch.backends.cudnn import benchmark

from UNET.model import UNet
from UNET.utils import create_dir, seeding


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)  # (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  # (512, 512, 3)
    return mask


if __name__ == "__main__":
    """seeding"""
    seeding(42)

    """folders"""
    create_dir("output")

    """load dataset"""
    test_x = sorted(glob("..\\new_data\\test\\*"))

    """Hyperparams"""
    checkpoint_path = "/UNET/files/gabor.tar"
    size = (1624, 1232)

    print(f"Test: {len(test_x)}")

    """load the checkpoint"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet()
    model = model.to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    torch.backends.cudnn.benchmark = True
    model.eval()

    time_taken = []

    for x in tqdm(test_x, total=len(test_x)):
        """extract the name"""
        name = x.split("\\")[-1].split('.')[0]

        """read images"""
        image = cv2.imread(x, cv2.IMREAD_COLOR)  # (512,512,3)
        # image.cv2.resize(image, size)
        x = np.transpose(image, (2, 0, 1))  # (3, 512, 512)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)  # (1, 3, 512, 512)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)


        with torch.no_grad():
            """predicting and calculating fps"""
            start_time = time.time()
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)
            total_time = time.time() - start_time
            time_taken.append(total_time)
            pred_y = pred_y[0].cpu().numpy()  # (1, 512, 512)
            pred_y = np.squeeze(pred_y, axis=0)  # (512, 512)
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)

        """saving masks"""
        pred_y = cv2.resize(pred_y, size)
        cv2.imwrite(f"output\\{name}.png", pred_y * 255)

    fps = 1 / np.mean(time_taken)
    print("FPS: ", fps)
