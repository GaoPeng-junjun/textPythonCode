import numpy as np


# Create the random array
# np.random.seed(100)
# arr = np.random.random([1, 3])/1e3
# np.set_printoptions(suppress=True, precision=6)  # precision is optional

# np.set_printoptions(threshold=6)
# arr = np.arange(15)
# np.set_printoptions(threshold=float('inf'))
#
# print(arr)

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# iris = np.genfromtxt(url, delimiter=',', dtype='object')
# # names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
#
# print(iris.shape)
#
# # np.set_printoptions(threshold=5)
# species = np.array([row[4] for row in iris])
# print(species)

iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0, 1, 2, 3])
print(iris_2d[:4])
