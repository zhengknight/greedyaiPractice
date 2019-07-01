#encoding=gbk
"""
KNN练习，K-Nearest Neighbor，K邻近算法，K为需要训练的超参数
"""
from sklearn import datasets  #导入sklearn的默认数据集用于练习
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

#加载datasets中的iris数据集
#iris数据集（只包含150个样本，每个样本有4个特征，共有3种分类）
iris = datasets.load_iris()
#print(iris.values())
#print(iris.items())
x = iris.data  #特征矩阵
y = iris.target   #标签
#print(x,y)
#划分训练集和测试集,如果不设置训练集和测试集的大小，默认训练集为数据集的25%
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=2001)
#print(len(y_test))
#准备训练模型,N=3
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train,y_train)

#使用训练模型进行预测
#方法一、使用numpy的非零计数函数count_nonzero
#correct = np.count_nonzero((clf.predict(x_test)==y_test)==True)
#print("模型在测试集上的正确率为:{:.5}".format(correct/len(y_test)))
#方法二、使用sklearn的方法
correct = accuracy_score(y_test, clf.predict(x_test))
print("模型在测试集上的正确率为%.3f" % correct)
#注意2种格式化输出方式，一种是{}.format，一种是%


