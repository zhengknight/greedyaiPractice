#encoding=gbk
"""
KNN练习，K-Nearest Neighbor，K邻近算法，K为需要训练的超参数
"""
from sklearn import datasets  #导入sklearn的默认数据集用于练习
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#加载datasets中的iris数据集
#iris数据集（只包含150个样本，每个样本有4个特征，共有3种分类）
iris = datasets.load_iris()
#print(iris.values())
#print(iris.items())
x = iris.data
y = iris.target
#print(x,y)
#划分训练集和测试集,如果不设置训练集和测试集的大小，默认训练集为数据集的25%
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=2001)
#print(len(y_test))
#准备训练模型,N=3
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train,y_train)


