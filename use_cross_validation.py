from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X = iris.data
y = iris.target

import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# knn = KNeighborsClassifier(n_neighbors=5)
# scores = cross_val_score(knn,X,y,cv=5,scoring=None) # cv =5 表示5折
# print(scores.mean())

k_range = range(1,31)
k_scores = []
# for k in k_range:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     scores = cross_val_score(knn, X, y, cv=5, scoring=None)  # cv =5 表示5折
#     k_scores.append(scores.mean())
#
# plt.plot(k_range,k_scores)
# plt.show()

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')  # cv =5 表示5折  用于classification
    # loss = -cross_val_score(knn,X,y,cv=10,scoring='neg_mean_squared_error')    # 用于regression
    k_scores.append(scores.mean())
    # k_scores.append(loss.mean())

plt.plot(k_range,k_scores)
plt.show()