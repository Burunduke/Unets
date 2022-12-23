import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import filters, color
from scipy import ndimage as ndi
import numpy as np


path = r'D:\Study\Python\unets\input\train\1.png'
path_1 = r"D:\Study\Python\unets\new_data\clahe\train\image\1_0.png"
fig, ax = plt.subplots(nrows=4, ncols=4)


image_color = cv2.imread(path, 1)
image_grey = cv2.imread(path, 0)
image2_color = cv2.imread(path_1, 1)
image2_grey = cv2.imread(path_1, 0)
size = (256, 256)
image_gay = np.stack([image2_grey, image2_grey, image2_grey])

#clahe
clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
cl1 = clahe.apply(image_grey)
cl1 = cv2.resize(cl1, size)
ax[1, 0].imshow(cl1)
ax[1, 0].set_title('Clahe')

#grey
x = image_grey
x = cv2.resize(x, size)
ax[2, 0].imshow(x)
ax[2, 0].set_title('grey')

b, g, r = cv2.split(image_color)

#b, g, r
b = cv2.resize(b, size)
ax[0, 1].imshow(b)
ax[0, 1].set_title('b')
g = cv2.resize(g, size)
ax[1, 1].imshow(g)
ax[1, 1].set_title('g')
r = cv2.resize(r, size)
ax[2, 1].imshow(r)
ax[2, 1].set_title('r')


#sobel
cv2.imshow("hui", image_grey)
edges = filters.sobel(image_grey)
cv2.imshow("hui2", edges)
edges = cv2.resize(edges, size)
ax[3, 1].imshow(edges)
ax[3, 1].set_title('Sobel edges')

#roberts
roberts = filters.roberts(image_grey)
roberts = cv2.resize(roberts, size)
ax[0, 2].imshow(roberts)
ax[0, 2].set_title('roberts')

#prewitt
prewitt = filters.prewitt(image_grey)
prewitt = cv2.resize(prewitt, size)
ax[1, 2].imshow(prewitt)
ax[1, 2].set_title('prewitt')


#sato
t0, t1 = filters.threshold_multiotsu(image_grey)
mask = (image_grey > t0)
vessels = filters.sato(image_grey, sigmas=range(1, 10)) * mask
vessels = cv2.resize(vessels, size)
ax[2, 2].imshow(vessels)
ax[2, 2].set_title('Sato vesselness')


#meijering
meijering = filters.meijering(image_grey)
meijering = cv2.resize(meijering, size)
ax[0, 3].imshow(meijering)
ax[0, 3].set_title('meijering')

#scharr
scharr = filters.scharr(image_grey)
scharr = cv2.resize(scharr, size)
ax[1, 3].imshow(scharr)
ax[1, 3].set_title('scharr')

#unsharp_mask
unsharp_mask = filters.unsharp_mask(image_grey)
unsharp_mask = cv2.resize(unsharp_mask, size)
ax[2, 3].imshow(unsharp_mask)
ax[2, 3].set_title('unsharp_mask')


for a in ax.ravel():
    a.axis('off')

plt.tight_layout()

plt.show()
