import numpy as np
from minisom import MiniSom
from matplotlib import pyplot as plt
import joblib
from matplotlib.cm import get_cmap

data = np.loadtxt('s1.txt')

input_dim = 2
output_dim = 15     # number of clusters

learning_rate = 0.1

som = MiniSom(1, 15, input_dim, sigma=0.5, learning_rate=0.5)

max_iter = 1000
q_error = []
t_error = []

min_val = data.min(axis=0)
max_val = data.max(axis=0)
normalized = (data - min_val) / (max_val - min_val)

som.train(data, 100000, verbose=True)

# for i in range(max_iter):
#     rand_i = np.random.randint(len(data))
#     som.update(data[rand_i], som.winner(data[rand_i]), i, max_iter)
#     q_error.append(som.quantization_error(data))
#     t_error.append(som.topographic_error(data))

cluster_labels = som.win_map(data)

print(cluster_labels)

cmap = get_cmap('tab20', output_dim)

som_filename = "trained_som_model.pkl"
joblib.dump(som, som_filename)

plt.figure(figsize=(8, 6))
for i, key in enumerate(cluster_labels.keys()):
    cluster = np.array(cluster_labels[key])
    color = cmap(i % output_dim)
    plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {key}',  color=color, alpha=0.7)

plt.title(f'SOM Clustering with {15} Clusters')
plt.legend()
plt.show()
