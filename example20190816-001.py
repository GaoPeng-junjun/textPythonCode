from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

#  Broadcasting 广播
# x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# v = np.array([1, 0, 1])
# print(x)
# print(v)
# print(x + v)

# opencv 读取图片
img = cv2.imread('picture/qianxun.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# cv2.imshow('src', img)
# cv2.imshow('gray', gray)
# cv2.imshow('RGB', img_rgb)
#
# cv2.waitKey()

# matplotlib 绘制图像
plt.subplot(1, 3, 1)
plt.imshow(img)

plt.subplot(1, 3, 2)
plt.imshow(gray)

plt.subplot(1, 3, 3)
plt.imshow(img_rgb)

plt.show()
#  Numpy: 提供了一个高性能的多维数组和基本工具来计算和操作这些数组
#  SciPy: 以Numpy为基础，提供了大量在numpy数组上运行的函数
# img = Image.open('picture/qianxun.jpg')
# print(img.format)
# print(img.size)
# print(img.mode)
# img.show()
#
# gray = img.convert('L')
# print(gray.mode)
# gray.show()
#
# arr = np.array(img)
# print(arr.shape)
# print(arr.dtype)
# print(arr)
#
#
# gray = np.array(gray)
# print(gray.shape)
# print(gray.dtype)
# print(gray)
#
# new_in = Image.fromarray(arr)
# new_in.save('picture/test.jpg')


