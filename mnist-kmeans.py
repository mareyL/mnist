import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from sklearn.cluster import KMeans 
from sklearn.pipeline import Pipeline

## import data

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape((60000, 784))
test_images = test_images.reshape((10000, 784))

# Normalize pixel values to be between -1 and 1
train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1

##kmeans implementation

def kmeans(X, k, iterations):
    
    n = X.shape[0]
    errors = []
    
    centroids = []
    
    for i in range(k):
        centroids.append(X[np.random.randint(0,n)])
        
    for j in range(iterations):
        
        
        clusterAssignement = []
        distance = []
        for x in X:
            temp_dist = []
            for c in centroids:
                temp_dist.append(np.linalg.norm(c - x))
            distance.append(temp_dist)
        for d in distance:
            clusterAssignement.append(d.index(min(d)))
        
        clusters = [[] for i in range(k)]
        for i, x in zip(clusterAssignement, X):
            clusters[i].append(x)
        centroids = [np.mean(cluster,axis = 0) for cluster in clusters]
        
        error = 0
        for i, centroid in zip(range(len(centroids)), centroids):
            for x in X[i]:
                error += np.linalg.norm(x - centroid)
        errors.append(error)
    
    return clusterAssignement,errors

clusterAssignement, errors = kmeans(test_images, 10, 100)
acc = 0
for i in range(len(clusterAssignement)):
    acc += clusterAssignement[i] == test_labels[i]
print("accuracy = ", acc/len(clusterAssignement))


## plot 100 examples


plt.figure(figsize=(20,20))
for index, (image, label) in enumerate(zip(train_images[0:100], clusterAssignement[0:100])):
    plt.subplot(5, 20, index + 1)
    plt.axis("off")
    plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
    plt.title(label, fontsize = 20)
    plt.show()



## comparison to the sklearn algorithm

pca = PCA(n_components=10)
kmeans = KMeans(n_clusters=10,n_init=1)
predictor = Pipeline([('pca', pca), ('kmeans', kmeans)])
predict = predictor.fit(test_images).predict(test_images)

acc = 0
for i in range(len(predict)):
    acc += predict[i] == test_labels[i]
print("accuracy = ", acc/len(predict))


plt.figure(figsize=(20,20))
for index, (image, label) in enumerate(zip(train_images[0:100], predict[0:100])):
    plt.subplot(5, 20, index + 1)
    plt.axis("off")
    plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
    plt.title(label, fontsize = 20)
    plt.show()