import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

try:
    X = np.loadtxt(r"D:\politeh\ai\lab7\data_clustering.txt", delimiter=',')
except ValueError:
    X = np.loadtxt(r"D:\politeh\ai\lab7\data_clustering.txt", delimiter=',')

num_clusters = 5

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X[:,0], X[:,1], marker='o', facecolors='none', edgecolors='black', s=80)
plt.title('Вхідні дані (з файлу)')

kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, random_state=0)
kmeans.fit(X)

step_size = 0.01
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max, step_size),
                             np.arange(y_min, y_max, step_size))

output = kmeans.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
output = output.reshape(x_vals.shape)

plt.subplot(1, 2, 2)
plt.imshow(output, interpolation='nearest',
           extent=(x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()),
           cmap=plt.cm.Paired, aspect='auto', origin='lower')

plt.scatter(X[:,0], X[:,1], marker='o', facecolors='none', edgecolors='black', s=80)

centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], marker='o', s=200, linewidths=4, color='black', facecolors='black')

plt.title('Результат K-Means')
plt.show()