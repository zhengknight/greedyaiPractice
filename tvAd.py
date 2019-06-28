#encoding=gbk
""""
���ӹ��Ч��Ԥ����ϰ
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("Advertising.csv")
#����ļ�ͷ
#print(data.head())
#����ļ�������
#print(data.columns)

#��ʾ����ɢ��ͼ
# plt.figure(figsize=(16,8))
# plt.scatter(data['TV'],data['sales'],c='black')
# plt.xlabel("Money spent on TV ads")
# plt.ylabel("Sales")
# plt.show()
#ѵ�����Իع�ģ��
x = data['TV'].values.reshape(-1,1)
y = data['sales'].values.reshape(-1,1)
#print(x)
reg = LinearRegression()
reg.fit(x,y)

print('a = {:.5}'.format(reg.coef_[0][0]))
print('b = {:.5}'.format(reg.intercept_[0]))
print("����ģ��Ϊ��Y={:.5}X+{:.5}".format(reg.coef_[0][0],reg.intercept_[0]))

#���ӻ�ѵ���õ����Իع�ģ��
predictions = reg.predict(x)
plt.figure(figsize=(16,8))
plt.scatter(data['TV'],data['sales'],c='black')
plt.plot(data['TV'],predictions,c='blue',linewidth=2)
plt.xlabel("Money spent on TV ads")
plt.ylabel("Sales")
plt.show()

#��Ԥ�⣬������һ��ȹ�˾��Ͷ��100������ôԤ�ڵ���������ǣ�
predictions_100 = reg.predict([[100]])
print("Ͷ��1�ڹ��ѣ�Ԥ������Ϊ��{:.5}".format(predictions_100[0][0]))