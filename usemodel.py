from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import datasets

X,y = datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=10)  # 自己生成数据

model = LinearRegression()
model.fit(X,y)
# plt.scatter(X,y)
# plt.show()
# 参数
print(model.coef_)
print(model.intercept_)
print(model.score(X,y))
