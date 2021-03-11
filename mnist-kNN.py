import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from sklearn.neighbors import KNeighborsClassifier
import time
import pickle
import sys

## import data

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape((60000, 784))
test_images = test_images.reshape((10000, 784))

# Normalize pixel values to be between -1 and 1
train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1

##kNN implementation

t = time.time()
neigh = KNeighborsClassifier(n_neighbors=10, weights='distance', leaf_size=100)
neigh.fit(train_images, train_labels)
print("fit time:", time.time()-t)
t = time.time()
result = neigh.predict(test_images)
print("predict time:", time.time()-t)

acc = 0
for i in range(len(test_images)):
    acc += result[i] == test_labels[i]
print("accuracy = ", acc/len(test_images))

print(sys.getsizeof(pickle.dumps(neigh)))
