import numpy as np

# 填充日期序列中缺乏值
# datas = np.arange(np.datetime64('2019-08-01'), np.datetime64('2019-08-19'), 2)
# print(datas)
# print(np.diff(datas))
# filled_in = np.array([np.arange(data, (data+d)) for data, d in zip(datas, np.diff(datas))]).reshape(-1)
#
# # for data, d in zip(datas, np.diff(datas)):
# #     print(data)
# #     print(data+d)
# #     print(np.arange(data, (data+d)))
# print(filled_in)
#
# out = np.hstack([filled_in, datas[-1]])
# print(out)


# 从给定的一维数组arr中，利用步进生成一个二维矩阵，窗口长度为4，步距为2
def gen_strides(a, stride_len=5, window_len=5):
    n_strides = ((a.size-window_len)//stride_len)+1
    return np.array([a[s:(s+window_len)] for s in np.arange(0, n_strides*stride_len, stride_len)])


arr = np.arange(15)
print(arr)
print(gen_strides(arr, stride_len=2, window_len=5))
