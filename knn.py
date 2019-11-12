from mlxtend.data import loadlocal_mnist

import numpy as np
from scipy import linalg as LA
from collections import Counter

def pca3(X):
  # Data matrix X, assumes 0-centered
  n, m = X.shape
  assert np.allclose(X.mean(axis=0), np.zeros(m))
  # Compute covariance matrix
  C = np.dot(X.T, X) / (n-1)
  # Eigen decomposition
  evals, evecs = LA.eigh(C)

  idx = np.argsort(evals)[::-1]
  evecs = evecs[:,idx]
  evals = evals[idx]
  evecs = evecs[:, :9]
  return evecs

  # Project X onto PC space
  # np.set_printoptions(precision=6)
  # print(eigen_vecs[0])
  # Size = np.dot(eigen_vecs.T, eigen_vecs)
  # size2 = Size.diagonal()
  # print(size2)
  # result = np.where(size2==np.amax(size2))
  # print(result)
  # X_pca = np.dot(X, eigen_vecs)
  # return np.dot(X,evecs)

X, y = loadlocal_mnist(
        images_path='C:/Users/SimoRyu/Documents/Freshman_2/ai/knn/train-images.idx3-ubyte',
        labels_path='C:/Users/SimoRyu/Documents/Freshman_2/ai/knn/train-labels.idx1-ubyte')


TestX, Testy = loadlocal_mnist(
            images_path='C:/Users/SimoRyu/Documents/Freshman_2/ai/knn/t10k-images.idx3-ubyte',
            labels_path='C:/Users/SimoRyu/Documents/Freshman_2/ai/knn/t10k-labels.idx1-ubyte')

print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
print('\n 1st row', X[0])

print(y)

X = X - X.mean(axis=0)

PCA_3dim = pca3(X)

Xred3 = np.dot(X, PCA_3dim)

TestX = TestX - TestX.mean(axis=0)
Testred3 = np.dot(TestX, PCA_3dim)

print(Testred3.shape[0])

def takeSecond(elem):
    return elem[1]

def kNN(k, TrainDat, TrainLabel, TestDat, TestLabel):
    ResultLabel = []
    correct = 0
    for i in range( int(TestDat.shape[0])):
        priqueue = []
        for j in range(TrainDat.shape[0]):
            dist = np.linalg.norm(TrainDat[j]-TestDat[i])
            Dup = (dist, TrainLabel[j])
            priqueue.append(Dup)
        priqueue = sorted(priqueue)

        neigh = [priqueue[t][1] for t in range(k)]

        if max(neigh, key = neigh.count) == TestLabel[i]:
            correct += 1
        if correct % 100 ==0:
            print(correct, i)

    return correct
y = kNN(8, X, y, TestX, Testy)


print(y)
