#encoding=gbk
"""
���ú��Ρ��¶����ݽ������ͻع���ϰ
��LinearRegressionΪOrdinary least squares Linear Regression(��ͨ��С�������Իع�)
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#��һ��������pandas��������
data = pd.read_csv('height.vs.temperature.csv')
#print(data.head())
#print(data['height'])
#print(data['height'].values)
#�ڶ�������ʾɢ��ͼ��ֱ�۹۲�һ����
"""
ɢ��ͼ����Ϊ1�����廭������Ҫ����Ϊ�����ߴ�2������x/y���ǩ
3���������ݵ�x��y����չʾ��ɫ����������4��ȫ���������show
"""
# plt.figure(figsize=(18,9))
# plt.xlabel("���θ߶�")
# plt.ylabel("����ֵ")
# plt.scatter(data['height'],data['temperature'],c='red')
# plt.show()

#��������ģ��ѵ��

#����fit�����������Ϊ���������Ҫ���ȶ����ݽ�����Ӧ�任
x = data['height'].values.reshape(-1,1)
y = data['temperature'].values.reshape(-1,1)
#ʵ����LinearRegression�࣬������fit����
reg = LinearRegression()
#reg.fit(data['height'].values,data['temperature'].values)ֱ�Ӵ�һά���飬�����ᱨ��
reg.fit(x,y)

#���Ĳ������ӻ�ѵ��ģ��
#��һ��ʹ��ѵ���õ�ģ����Ԥ��yֵ
predict = reg.predict(x)

plt.figure(figsize=(18,9))
plt.xlabel("���θ߶�")
plt.ylabel("����ֵ")
plt.scatter(data['height'],data['temperature'],c='red')
plt.plot(data['height'],predict,c='blue',linewidth=2)
plt.show()

#���岽������ѵ���õ�ģ��Ԥ��8000�׺��θ߶ȣ�����ֵ
predict_8000 = reg.predict([[8000]])
print("����Ԥ��ģ�ͼ���ó�������8000�׸߶ȣ�����ֵΪ:{:.5}".format(predict_8000[0][0]))



