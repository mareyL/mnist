import tensorflow as tf
from sklearn import svm, metrics
from sklearn import model_selection
import pandas as pd
import time
import pickle
import sys

# load MNIST dataset from keras library
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# normalize pixel values to be between -1 and 1
train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1

# reshaping the data
train_images = train_images.reshape((60000, 784))
test_images = test_images.reshape((10000, 784))

def getmodelsize(classifier):
    p = pickle.dumps(classifier)
    print("Size of linear SVM model is %s Bytes",sys.getsizeof(p))

def svmlinearmodel():
    # a linear support vector classifier
    classifier = svm.SVC(kernel ='linear')

    # Learning Phase
    start_time = time.time()
    classifier.fit(train_images, train_labels)
    print("Fit time for linear SVM is --- %s seconds ---" % (time.time() - start_time))

    # Prediction phase
    start_time = time.time()
    predicted = classifier.predict(test_images)
    print("Prediction time for linear SVM is --- %s seconds ---" % (time.time() - start_time))

    # classification report
    print("\n")
    print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(test_labels, predicted)))

    # accuracy
    print("accuracy:", metrics.accuracy_score(test_labels, predicted), "\n")
    # confusion matrix
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_labels, predicted))

    # model size
    getmodelsize(classifier)


def svmrbfdefaultmodel():

    # a non-linear support vector classifier using rbf kernel, C=1, default value of gamma
    non_linear_model = svm.SVC(kernel='rbf')

    # learning phase
    start_time = time.time()
    non_linear_model.fit(train_images, train_labels)
    print("Fit time for non-linear rbf kernal is --- %s seconds ---" % (time.time() - start_time))

    # prediction phase
    start_time = time.time()
    nonlinear_predicted = non_linear_model.predict(test_images)
    print("Prediction time for non-linear rbf kernal is --- %s seconds ---" % (time.time() - start_time))

    # classification report
    print("\n")
    print("Classification report for non-linear classifier %s:\n%s\n" % (non_linear_model, metrics.classification_report(test_labels, nonlinear_predicted)))
    print("accuracy:", metrics.accuracy_score(test_labels, nonlinear_predicted), "\n")

    # confusion matrix
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_labels, nonlinear_predicted))

    # model size
    getmodelsize(non_linear_model)

def svmbrbf_bestmodel():
    # using rbf kernel
    best_non_linear_model = svm.SVC(C=5, gamma=0.01, kernel='rbf')

    # training phase
    start_time = time.time()
    best_non_linear_model.fit(train_images, train_labels)
    print("Fit time for best non-linear rbf kernal is --- %s seconds ---" % (time.time() - start_time))

    # prediction phase
    start_time = time.time()
    nonlinear_predicted = best_non_linear_model.predict(test_images)
    print("Prediction time for best non-linear rbf kernal is --- %s seconds ---" % (time.time() - start_time))

    # classification report
    print("\n")
    print("Classification report for non-linear classifier %s:\n%s\n" % (best_non_linear_model, metrics.classification_report(test_labels, nonlinear_predicted)))
    print("accuracy:", metrics.accuracy_score(test_labels, nonlinear_predicted), "\n")

    # confusion matrix
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_labels, nonlinear_predicted))

    # model size
    getmodelsize(best_non_linear_model)


def tuneparameter():

    #we will use only half of the data, otherwise it takes way too much time to tune parameters
    train_images = train_images[:30000]
    train_labels = train_labels[:30000]

    # creating a KFold object with 3 splits
    folds = model_selection.KFold(n_splits = 3, shuffle = True, random_state = 10)

    # specify range of hyperparameters
    # Set the parameters by cross-validation
    hyper_params = [ {'gamma': [1e-2, 1e-3, 1e-4],
                         'C': [5,10]}]

    # specify model
    model = svm.SVC(kernel="rbf")

    # set up GridSearchCV()
    model_cv = model_selection.GridSearchCV(estimator = model,
                            param_grid = hyper_params,
                            scoring= 'accuracy',
                            cv = folds,
                            verbose = 1,
                            return_train_score=True)

    # fit the model
    model_cv.fit(train_images, train_labels)

    # cv results
    cv_results = pd.DataFrame(model_cv.cv_results_)

    # converting C to numeric type for plotting on x-axis
    cv_results['param_C'] = cv_results['param_C'].astype('int')

    # printing the optimal accuracy score and hyperparameters
    best_score = model_cv.best_score_
    best_hyperparams = model_cv.best_params_

    print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))


if __name__ == '__main__':

    print(" --- Classification using linear kernel SVM ---")
    print("\n")
    svmlinearmodel()

    print(" --- Classification using non-linear SVM with rbf kernel ---")
    print("\n")
    svmrbfdefaultmodel()

    # use only to get the best value of parameters. Dont use if already known. It takes too much time ~8 hrs to tune c and gamma
    #tuneparameter()

    print(" --- Classification using best non-linear rbf kernel SVM with C=5, gamma=0.01 ---")
    print("\n")
    svmbrbf_bestmodel()
