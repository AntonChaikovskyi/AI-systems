import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

try:
    X = np.loadtxt(r"D:\politeh\ai\lab7\data_clustering.txt", delimiter=',')
except ValueError:
    X = np.loadtxt(r"D:\politeh\ai\lab7\data_clustering.txt", delimiter=',')

bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=len(X))

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)

labels = ms.labels_
cluster_centers = ms.cluster_centers_
n_clusters_ = len(np.unique(labels))

print(f"Кількість знайдених кластерів: {n_clusters_}")

plt.figure()
colors = 10 * ['r', 'g', 'b', 'c', 'k', 'y', 'm']
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)

plt.title(f'Mean Shift: знайдено {n_clusters_} кластерів')
plt.show()