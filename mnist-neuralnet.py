import tensorflow as tf
import larq as lq
import pickle
import sys
import time

# load MNIST dataset from keras library
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# reshaping the input
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# normalize pixel values to be between -1 and 1
train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1

def getmodelsize(classifier):
    p = pickle.dumps(classifier)
    print("Size of linear SVM model is %s Bytes",sys.getsizeof(p))


def twolayer_NN():

    # create a model by adding layers
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3),
                                use_bias=False,
                                input_shape=(28, 28, 1)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, use_bias=False))
    model.add(tf.keras.layers.Activation("softmax"))

    print("------------Model size of 2 layer Convolutional network--------")
    lq.models.summary(model)

    # compile the model with necessary parameters like optimizers and loss functions to be used
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    # training phase: training the model with training data
    start_time = time.time()
    model.fit(train_images, train_labels, batch_size=64, epochs=10)
    print("Fit time for 2 layer neural network is --- %s seconds ---" % (time.time() - start_time))

    # prediction phase
    start_time = time.time()
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("Prediction/evaluation time for 2 layer neural network is --- %s seconds ---" % (time.time() - start_time))

    # accuracy
    print(f"Test accuracy {test_acc * 100:.2f} %")


def threelayer_NN():

    # create a model by adding layers (Added 1 extra hidden fully connected layer)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3),
                                use_bias=False,
                                input_shape=(28, 28, 1)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, use_bias=False))
    model.add(tf.keras.layers.Dense(10, use_bias=False))
    model.add(tf.keras.layers.Activation("softmax"))

    print("------------Model size of 3 layer Convolutional network--------")
    lq.models.summary(model)

    # compile the model with necessary parameters like optimizers and loss functions to be used
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    # training phase: training the model with training data
    start_time = time.time()
    model.fit(train_images, train_labels, batch_size=64, epochs=10)
    print("Fit time for 3 layer neural network is --- %s seconds ---" % (time.time() - start_time))

    # prediction phase
    start_time = time.time()
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("Prediction/evaluation time for 3 layer neural network is --- %s seconds ---" % (time.time() - start_time))

    # accuracy
    print(f"Test accuracy {test_acc * 100:.2f} %")


if __name__ == '__main__':
    print("------------Classifying using 2 layer Convolutional network--------")
    twolayer_NN()
    print("\n")
    print("------------Classifying using 3 layer Convolutional network--------")
    threelayer_NN()
