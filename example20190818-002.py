import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# 加载数据集
fruits_df = pd.read_table('模型评价指标/fruit_data_with_colors.txt')
print(fruits_df.head())

# 划分数据集
X = fruits_df[['mass', 'width', 'height', 'color_score']]
y = fruits_df['fruit_label'].copy()

# 转换为二分类问题
y[y != 1] = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=0)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# 准确率
print('准确率：{:.3f}'.format(accuracy_score(y_test, y_pred)))

# 精确率
print('精确率：{:.3f}'.format(precision_score(y_test, y_pred)))

# 召回率
print('召回率：{:.3f}'.format(recall_score(y_test, y_pred)))

# F1值
print('F1值：{:.3f}'.format(f1_score(y_test, y_pred)))

# PR曲线
precision, recall, _ = precision_recall_curve(y_test, y_pred)
print('AP值：{:.3f}'.format(average_precision_score(y_test, y_pred)))

# # ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_pred)
print('AUC值：{:.3f}'.format(roc_auc_score(y_test, y_pred)))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure()
plt.grid(False)
plt.imshow(cm, cmap=plt.cm.Blues)
plt.colorbar()
plt.show()

# iris数据集上使用混淆矩阵查看结果
# 加载数据
iris = load_iris()

X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=0)

# 模型训练预测
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# 获取混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 归一化处理，得到每个分类的准确率
cm_norm = cm / cm.sum(axis=1)

print('未归一化')
print(cm)

print('归一化')
print(cm_norm)

plt.figure()
plt.grid(False)
plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
plt.show()
