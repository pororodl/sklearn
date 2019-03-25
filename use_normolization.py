from sklearn import preprocessing
import numpy as np
from sklearn.datasets.samples_generator import make_classification
import matplotlib.pyplot as plt
from sklearn import datasets
a = np.array([[10,2.7, 3.6],
              [-100, 5, 2],
              [120,20,40]],dtype = np.float64)

# print(a)
# print(preprocessing.scale(a))
X,y = make_classification(n_samples=300,n_features=2,n_redundant=0,
                          n_informative=2,random_state=22,n_clusters_per_class=1,
                          scale=100)
print(type(X))
print(np.shape(X))
print(np.shape(y))
# plt.scatter(X[:,0],X[:1],c=y)
# plt.show()
# X = preprocessing