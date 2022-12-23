import os

import matplotlib.pyplot as plt
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio.v2 as imageio
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from sklearn.model_selection import train_test_split
from skimage import filters

"""создаем директорию если ее нету"""


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


"""загружаем данные"""


def load_data(path):
    train_x = glob(os.path.join(path, "train", "*[0-9].png"))
    train_y = []
    for train in train_x:
        train_y.append(train.replace(".png", "_mask.png"))

    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
    return (train_x, train_y), (test_x, test_y)


def augument_data(images, masks, mode, save_path):
    size = (256, 256)
    print(mode)

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):

        """выводим имя"""
        name = x.split("\\")[-1].split('.')[0]

        """считываем картиночку и маску"""

        y = imageio.imread(y)

        match mode:
            case "clahe":
                x = cv2.imread(x, 0)
                clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
                cl1 = clahe.apply(x)
                x = cl1
            case "grey":
                x = cv2.imread(x, 0)
            case "dog":
                x = cv2.imread(x, 0)
                low_sigma = cv2.GaussianBlur(x, (3, 3), 0)
                high_sigma = cv2.GaussianBlur(x, (5, 5), 0)
                x = low_sigma - high_sigma
            case "r":
                x = cv2.imread(x, 1)
                b, g, r = cv2.split(x)
                x = r
            case "g":
                x = cv2.imread(x, 1)
                b, g, r = cv2.split(x)
                x = g
            case "b":
                x = cv2.imread(x, 1)
                b, g, r = cv2.split(x)
                x = b
            case "sobel":
                x = cv2.imread(x, 0)
                x = filters.sobel(x)
                x *= 255
            case "roberts":
                x = cv2.imread(x, 0)
                x = filters.roberts(x)
                x *= 255
            case "prewitt":
                x = cv2.imread(x, 0)
                x = filters.prewitt(x)
                x *= 255
            case "sato":
                x = cv2.imread(x, 0)
                t0, t1 = filters.threshold_multiotsu(x)
                mask = (x > t0)
                x = filters.sato(x, sigmas=range(1, 10)) * mask
                x *= 255
            case "meijering":
                x = cv2.imread(x, 0)
                x = filters.meijering(x)
                x *= 255
            case "scharr":
                x = cv2.imread(x, 0)
                x = filters.scharr(x)
                x *= 255
            case "unsharp_mask":
                x = cv2.imread(x, 0)
                x = filters.unsharp_mask(x)
                x *= 255
            case "multidim":
                x1 = cv2.imread(x, 0)
                x1 = filters.unsharp_mask(x1)
                x1 *= 255
                x2 = cv2.imread(x, 0)
                clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
                cl1 = clahe.apply(x2)
                x2 = cl1
                x3 = cv2.imread(x, 1)
                b, g, r = cv2.split(x3)
                x3 = g
                x = np.stack([x1, x2, x3])
                x = np.transpose(x, (1, 2, 0))
            case _:
                x = cv2.imread(x, 1)


        aug = HorizontalFlip(p=1.0)
        augumented = aug(image=x, mask=y)
        x1 = augumented["image"]
        y1 = augumented["mask"]

        aug = VerticalFlip(p=1.0)
        augumented = aug(image=x, mask=y)
        x2 = augumented["image"]
        y2 = augumented["mask"]

        aug = Rotate(limit=45, p=1.0)
        augumented = aug(image=x, mask=y)
        x3 = augumented["image"]
        y3 = augumented["mask"]

        X = [x, x1, x2, x3]
        Y = [y, y1, y2, y3]


        index = 0
        for i, m in zip(X, Y):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            tmp_img_name = f"{name}_{index}.png"
            tmp_msk_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "image", tmp_img_name)
            mask_path = os.path.join(save_path, "mask", tmp_msk_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1


if __name__ == "__main__":
    np.random.seed(42)

    """создаем новые данные из старых"""
    data_path = "D:\\Study\\Python\\unet\\input"
    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    """сколько разных файлов"""
    print(f"\nTrain: {len(train_x)} {len(train_y)}")
    print(f"Test: {len(test_x)} {len(test_y)}")

    augs = ["original", "clahe", "grey", 'r', 'g', 'b', 'sobel', "roberts", 'prewitt', 'sato',
            'meijering', 'scharr', 'unsharp_mask']
    augs_2 = ['sobel', "roberts", 'prewitt', 'sato', 'meijering', 'scharr', 'unsharp_mask']
    augs_3 = ["multidim"]
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    for mode in augs_3:
        create_dir(f"new_data\\{mode}\\train\\image")
        create_dir(f"new_data\\{mode}\\train\\mask")
        create_dir(f"new_data\\{mode}\\valid\\image")
        create_dir(f"new_data\\{mode}\\valid\\mask")

        """Из одной картиночки делаем 4 для тренировки, тестовым просто размер меняем"""
        augument_data(train_x, train_y, mode, save_path=f"new_data\\{mode}\\train\\")
        augument_data(test_x, test_y, mode, save_path=f"new_data\\{mode}\\valid\\")
