#encoding=gbk
"""
KNN��ϰ��K-Nearest Neighbor��K�ڽ��㷨��KΪ��Ҫѵ���ĳ�����
"""
from sklearn import datasets  #����sklearn��Ĭ�����ݼ�������ϰ
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

#����datasets�е�iris���ݼ�
#iris���ݼ���ֻ����150��������ÿ��������4������������3�ַ��ࣩ
iris = datasets.load_iris()
#print(iris.values())
#print(iris.items())
x = iris.data  #��������
y = iris.target   #��ǩ
#print(x,y)
#����ѵ�����Ͳ��Լ�,���������ѵ�����Ͳ��Լ��Ĵ�С��Ĭ��ѵ����Ϊ���ݼ���25%
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=2001)
#print(len(y_test))
#׼��ѵ��ģ��,N=3
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train,y_train)

#ʹ��ѵ��ģ�ͽ���Ԥ��
#����һ��ʹ��numpy�ķ����������count_nonzero
#correct = np.count_nonzero((clf.predict(x_test)==y_test)==True)
#print("ģ���ڲ��Լ��ϵ���ȷ��Ϊ:{:.5}".format(correct/len(y_test)))
#��������ʹ��sklearn�ķ���
correct = accuracy_score(y_test, clf.predict(x_test))
print("ģ���ڲ��Լ��ϵ���ȷ��Ϊ%.3f" % correct)
#ע��2�ָ�ʽ�������ʽ��һ����{}.format��һ����%


