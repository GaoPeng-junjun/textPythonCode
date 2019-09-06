import numpy as np

a = np.array([1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0])
# a = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# print(type(a))
# print(type(*np.where(a != 0)))
print(np.append(a, 1))
print(np.append(1, a))
print(np.where((np.append(a, 1) + np.append(1, a) == 1))[0])
print(np.delete(np.where((np.append(a, 1) + np.append(1, a) == 1))[0], -1))
print(np.diff(np.delete(np.where((np.append(a, 1) + np.append(1, a) == 1))[0], 0)))

max_length = np.max(np.diff(np.delete(np.where((np.append(a, 1) + np.append(1, a) == 1))[0], 0)))
print(max_length)
