###############################################
#          矩阵编程:矢量化、广播
###############################################


import numpy as np
import matplotlib.pyplot as plt


# sample = np.random.normal(loc=[2., 20.], scale=[1., 3.5], size=(3, 2))
# print(sample)
#
# print(sample.mean(axis=0))
# print(sample.mean(axis=1))
# print(sample.mean(axis=1)[:, None])
#
# print(sample-sample.mean(axis=0))
# print(sample-sample.mean(axis=1)[:, None])

# 求三角形的中心坐标
# tri = np.array([[1, 1],
#                [3, 1],
#                [2, 3]])
# centroid = tri.mean(axis=0)
#
# trishape = plt.Polygon(tri, edgecolor='r', alpha=0.2, lw=5)
# _, ax = plt.subplots(figsize=(4, 4))
# ax.add_patch(trishape)
# ax.set_ylim([.5, 3.5])
# ax.set_xlim([.5, 3.5])
# ax.scatter(*centroid, color='g', marker='D', s=70)
# ax.scatter(*tri.T, color='b',  s=70)
# plt.show()

# K-means迭代
def get_labels(X, centroids) -> np.ndarray:
    return np.argmin(np.linalg.norm(X - centroids[:, None], axis=2), axis=0)


X = np.repeat([[5, 5], [10, 10]], [5, 5], axis=0)
X = X + np.random.randn(*X.shape)
centroids = np.array([[5, 5], [10, 10]])
print(X)
print(centroids[:, None])
print(X-centroids[:, None])

# labels = get_labels(X, centroids)
# c1, c2 = ['#bc13fe', '#be0119']
# llim, ulim = np.trunc([X.min() * 0.9, X.max() * 1.1])
# _, ax = plt.subplots(figsize=(5, 5))
# ax.scatter(*X.T, c=np.where(labels, c2, c1), alpha=0.4, s=80)
# ax.scatter(*centroids.T, c=[c1, c2], marker='s', s=95, edgecolor='yellow')
# ax.set_ylim([llim, ulim])
# ax.set_xlim([llim, ulim])
# ax.set_title('One K-Means Iteration: Predicted Classes')
# plt.show()
