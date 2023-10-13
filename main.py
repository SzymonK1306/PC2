import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt  # For visualization (optional)
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import to_categorical
from sklearn.cluster import KMeans

data = np.loadtxt('data_set.txt')

X_train, X_test = train_test_split(data, test_size=0.3, random_state=42)
X_test, X_valid = train_test_split(X_test, test_size=0.5, random_state=42)

x_train = X_train[:, :2]
y_train = X_train[:, 2]

x_test = X_test[:, :2]
y_test = X_test[:, 2]

x_valid = X_valid[:, :2]
y_valid = X_valid[:, 2]

# Normalize features
min_val = x_train.min(axis=0)
max_val = x_valid.max(axis=0)
x_train = (x_train - min_val) / (max_val - min_val)
x_valid = (x_valid - min_val) / (max_val - min_val)
x_test = (x_test - min_val) / (max_val - min_val)

number_of_class = 15

input_dim = x_train.shape[1]  # Dimension of input data

n_clusters = len(np.unique(y_train))  # Number of clusters

y_train = to_categorical(y_train, num_classes=n_clusters)
y_test = to_categorical(y_test, num_classes=n_clusters)
y_valid = to_categorical(y_valid, num_classes=n_clusters)

# Define a simple neural network
input_layer = Input(shape=(input_dim,))
hidden_layer = Dense(64, activation='relu')(input_layer)  # You can adjust the number of units and layers
output_layer = Dense(n_clusters, activation='softmax')(hidden_layer)

model = Model(input_layer, output_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model with a validation set
model.fit(
    x_train,
    y_train,
    epochs=1000,
    batch_size=32,
    validation_data=(x_valid, y_valid)
)

# Evaluating the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)

# Predicting clusters for new data
# new_data = np.array([[new_cordinate_x, new_cordinate_y]])
# new_data = (new_data - min_val) / (max_val - min_val)
# cluster_probabilities = model.predict(new_data)
# predicted_cluster_label = np.argmax(cluster_probabilities)

print(f"Test Accuracy: {test_accuracy}")

# Save the model
model.save("cluster_model.h5")

# print(f"Predicted Cluster Label: {predicted_cluster_label}")


#-------------------------------------------------------------------
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(64, activation='tanh', input_shape=(x_train.shape[1],)),
#     tf.keras.layers.Dense(32, activation='tanh'),
#     tf.keras.layers.Dense(1, activation='softmax')
# ])
#
# model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=100000, batch_size=200, validation_data=(x_valid, y_valid))
#
# test_loss, test_accuracy = model.evaluate(x_test, y_test)
# print(f"Test accuracy: {test_accuracy}")
#----------------------------------------------------------------
