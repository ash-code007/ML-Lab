import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('ex.csv')
X= data.iloc[:,1:3].values

print('Graph for whole dataset')
plt.scatter(X[:,0], X[:,1], c='black', s=60)
plt.show()

from sklearn.cluster import KMeans
kmeans = KMeans(2, random_state=0)
kmeans.fit(X)
labels=kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, s=60);
print('Graph using Kmeans Algorithm')
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200)
plt.show()
#gmm demo
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=2)
gmm.fit(X)
labels = gmm.predict(X)

probs = gmm.predict_proba(X)
size = 10 * probs.max(1) ** 3
print('Graph using EM Algorithm')
plt.scatter(X[:, 0], X[:, 1], c=labels, s=size)
plt.show()