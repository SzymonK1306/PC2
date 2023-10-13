from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt  # For visualization (optional)
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import to_categorical
from sklearn.cluster import KMeans
from keras.models import load_model

loaded_model = load_model("cluster_model.h5")

# Example of using the loaded model for prediction
new_data = np.loadtxt('s1.txt')
control = np.loadtxt('data_set.txt')
min_val = new_data.min(axis=0)
max_val = new_data.max(axis=0)
normalized = (new_data - min_val) / (max_val - min_val)
# labels = [1 for i in range(5000)]
labels = []
cluster_probabilities = loaded_model.predict(normalized)
predicted_cluster_label = np.argmax(cluster_probabilities, axis=1)
labels.append(predicted_cluster_label)


# KMEANS
kmeans = KMeans(15)
kmeans.fit(new_data)

labels = kmeans.labels_
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.scatter(new_data[:, 0], new_data[:, 1], c=predicted_cluster_label, label="Sieciuch")
ax1.set_title('Sieciuch')
ax2.scatter(control[:, 0], control[:, 1], c=control[:, 2], label="k-means")
ax2.set_title('k-means')
plt.show()
