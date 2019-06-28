#encoding=gbk
"""
利用海拔、温度数据进行线型回归练习
类LinearRegression为Ordinary least squares Linear Regression(普通最小二乘线性回归)
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#第一步，利用pandas读入数据
data = pd.read_csv('height.vs.temperature.csv')
#print(data.head())
#print(data['height'])
#print(data['height'].values)
#第二步，显示散点图，直观观察一下下
"""
散点图步骤为1、定义画布，主要参数为画布尺寸2、设置x/y轴标签
3、给出数据点x、y及其展示颜色等属性设置4、全部设置完毕show
"""
# plt.figure(figsize=(18,9))
# plt.xlabel("海拔高度")
# plt.ylabel("气温值")
# plt.scatter(data['height'],data['temperature'],c='red')
# plt.show()

#第三步，模型训练

#由于fit方法传入参数为矩阵，因此需要首先对数据进行相应变换
x = data['height'].values.reshape(-1,1)
y = data['temperature'].values.reshape(-1,1)
#实例化LinearRegression类，调用其fit方法
reg = LinearRegression()
#reg.fit(data['height'].values,data['temperature'].values)直接传一维数组，本语句会报错
reg.fit(x,y)

#第四步，可视化训练模型
#第一步使用训练好的模型来预测y值
predict = reg.predict(x)

plt.figure(figsize=(18,9))
plt.xlabel("海拔高度")
plt.ylabel("气温值")
plt.scatter(data['height'],data['temperature'],c='red')
plt.plot(data['height'],predict,c='blue',linewidth=2)
plt.show()

#第五步，利用训练好的模型预测8000米海拔高度，气温值
predict_8000 = reg.predict([[8000]])
print("根据预测模型计算得出，海拔8000米高度，气温值为:{:.5}".format(predict_8000[0][0]))



