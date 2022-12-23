import matplotlib.pyplot as plt
from skimage import data, filters
import cv2

path = r'D:\Study\Python\unets\input\train\1.png'
path_1 = r"D:\Study\Python\unets\new_data\clahe\train\image\1_0.png"
size = (256, 256)

image_color = cv2.imread(path, 1)
image_color = cv2.resize(image_color, size)

image_grey = cv2.imread(path, 0)
clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
image_grey = clahe.apply(image_grey)
image_grey = cv2.resize(image_grey, size)


image2_color = cv2.imread(path_1, 1)

image2_grey = cv2.imread(path_1, 0)


fig, ax = plt.subplots(nrows=2, ncols=2)

ax[0, 0].imshow(image_color)
ax[0, 0].set_title('image_color')

ax[1, 0].imshow(image_grey)
ax[1, 0].set_title('image_grey')

ax[0, 1].imshow(image2_color)
ax[0, 1].set_title('image2_color')

ax[1, 1].imshow(image2_grey)
ax[1, 1].set_title('image2_grey')

for a in ax.ravel():
    a.axis('off')
plt.tight_layout()
plt.show()
