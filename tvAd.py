#encoding=gbk
""""
电视广告效果预测练习
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("Advertising.csv")
#输出文件头
#print(data.head())
#输出文件列名称
#print(data.columns)

#显示数据散点图
# plt.figure(figsize=(16,8))
# plt.scatter(data['TV'],data['sales'],c='black')
# plt.xlabel("Money spent on TV ads")
# plt.ylabel("Sales")
# plt.show()
#训练线性回归模型
x = data['TV'].values.reshape(-1,1)
y = data['sales'].values.reshape(-1,1)
#print(x)
reg = LinearRegression()
reg.fit(x,y)

print('a = {:.5}'.format(reg.coef_[0][0]))
print('b = {:.5}'.format(reg.intercept_[0]))
print("线性模型为：Y={:.5}X+{:.5}".format(reg.coef_[0][0],reg.intercept_[0]))

#可视化训练好的线性回归模型
predictions = reg.predict(x)
plt.figure(figsize=(16,8))
plt.scatter(data['TV'],data['sales'],c='black')
plt.plot(data['TV'],predictions,c='blue',linewidth=2)
plt.xlabel("Money spent on TV ads")
plt.ylabel("Sales")
plt.show()

#做预测，假设下一年度公司想投入100百万，那么预期的销量如何那？
predictions_100 = reg.predict([[100]])
print("投入1亿广告费，预计销量为：{:.5}".format(predictions_100[0][0]))