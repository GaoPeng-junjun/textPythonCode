from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib import stride_tricks


url = ('https://www.history.navy.mil/bin/imageDownload?image=/'
       'content/dam/nhhc/our-collections/photography/images/'
       '80-G-410000/80-G-416362&rendition=cq5dam.thumbnail.319.319.png')

img = io.imread(url, as_gray=True)
fig, ax = plt.subplots()
ax.imshow(img, cmap='gray')
ax.grid(False)
plt.title('img')
plt.show()

size = 10
m, n = img.shape
mm, nn = m - size + 1, n - size + 1
patch_means = np.empty((mm, nn))
for i in range(mm):
       for j in range(nn):
              patch_means[i, j] = img[i:(i+size), j:(j+size)].mean()

_, ax = plt.subplots()
ax.imshow(patch_means, cmap='gray')
ax.grid(False)
plt.title('patche_mean')
plt.show()

shape = (img.shape[0] - size + 1, img.shape[1] - size + 1, size, size)
strides = 2 * img.strides
patches = stride_tricks.as_strided(img, shape=shape, strides=strides)
# print(patches.shape)
patches_mean = patches.mean(axis=(-1, -2))
_, ax = plt.subplots()
ax.imshow(patches_mean, cmap='gray')
ax.grid(False)
plt.title('patches_mean')
plt.show()
