import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation

companies = ['Apple', 'Google', 'Microsoft', 'Amazon', 'Exxon', 'Chevron', 'Valero', 'Coca-Cola', 'Pepsi', 'Boeing', 'Lockheed', 'Ford', 'Toyota', 'Honda']

np.random.seed(42)
variation_data = []
variation_data.extend([np.random.normal(1.5, 0.2, 100) for _ in range(4)]) 
variation_data.extend([np.random.normal(-0.5, 0.3, 100) for _ in range(3)]) 
variation_data.extend([np.random.normal(0.1, 0.1, 100) for _ in range(2)]) 
variation_data.extend([np.random.normal(0.8, 0.4, 100) for _ in range(2)]) 
variation_data.extend([np.random.normal(-0.2, 0.2, 100) for _ in range(3)])
variation_data = np.array(variation_data)

model = AffinityPropagation(damping=0.5, random_state=0)
model.fit(variation_data)
labels = model.labels_
n_clusters_ = len(model.cluster_centers_indices_)

print(f'Знайдено кластерів: {n_clusters_}')
for i in range(n_clusters_):
    members = [companies[j] for j in range(len(companies)) if labels[j] == i]
    print(f"Кластер {i+1}: {', '.join(members)}")

plt.figure(figsize=(10, 6))
for i in range(n_clusters_):
    cluster_points = variation_data[labels == i].mean(axis=1)
    plt.scatter([i]*len(cluster_points), cluster_points, s=100, label=f'Clust {i+1}')
    for m, y in zip([companies[j] for j in range(len(companies)) if labels[j] == i], cluster_points):
        plt.text(i+0.1, y, m)
plt.title("Affinity Propagation (Ринок акцій)")
plt.show()