############################################
#             KNN简单练习
############################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from ml_visualization import plot_fruit_knn
import time

# 加载数据集
fruits_df = pd.read_table('knn/kNN_codes/fruit_data_with_colors.txt')

# 数据预览
# print(fruits_df.head())

# print('样本个数：', len(fruits_df))
# sns.countplot(fruits_df['fruit_name'], label="Count")
# plt.show()

# 创建目标标签和名称的字典
fruit_name_dict = dict(zip(fruits_df['fruit_label'], fruits_df['fruit_name']))
# print(fruit_name_dict)

# 划分数据集
X = fruits_df[['mass', 'width', 'height', 'color_score']]
y = fruits_df['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=0)
print('数据集样本数：{}，训练集样本数：{}，测试集样本数：{}'.format(len(X), len(X_train), len(X_test)))

# 查看数据集
# sns.pairplot(data=fruits_df, hue='fruit_name', vars=['mass', 'width', 'height', 'color_score'])
# plt.show()

label_color_dict = {1: 'red', 2: 'green', 3: 'blue', 4: 'yellow'}
colors = list(map(lambda label: label_color_dict[label], y_train))

fig1 = plt.figure(1)
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(X_train['width'], X_train['height'], X_train['color_score'], c=colors, marker='o', s=100)
ax.set_xlabel('width')
ax.set_ylabel('height')
ax.set_zlabel('color_score')
plt.draw()
plt.pause(2)
plt.close(fig1)


# 建立kNN模型
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
# 测试模型
y_pred = knn.predict(X_test)
print('预测标签：', y_pred)

acc = accuracy_score(y_test, y_pred)
print('准确率：', acc)

# 查看k值对结果的影响
k_range = range(1, 20)
acc_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc_scores.append(knn.score(X_test, y_test))

fig2 = plt.figure(2)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.plot(k_range, acc_scores, marker='o')
plt.xticks([0, 5, 11, 15, 21])
plt.draw()
plt.pause(2)
plt.close(fig2)


# 只查看width和height两列特征
plot_fruit_knn(X_train, y_train, 1, 3)
plot_fruit_knn(X_train, y_train, 5, 4)
plot_fruit_knn(X_train, y_train, 10, 5)

