import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

np.random.seed(12)

# Hyperparameter
K = 8

# TODO: Implement a function to compute the Euclidean distance between two data points.
def euclidean_distance(x1, x2):
    """
    Compute the Euclidean distance between two data points.

    Parameters:
    - x1, x2 (np.ndarray) (n, d): two data points to compute the distance.

    Returns:
    - float: the Euclidean distance between point1 and point2.
    """
    # TODO: First, compute the squared distance between two points
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    squared_distance = np.sum((x1 - x2) ** 2)

    # TODO: Second, compute the square root of the squared distance
    return float(np.sqrt(squared_distance))

# TODO: Use KNN to classify the data points in the test set.

def KNN_predict(x_train, y_train, x_test, k):
    ''' 
    Predict the labels for x_test using KNN.

    Parameters:
    - x_train (np.ndarray) (n, d): training data
    - y_train (np.ndarray) (n,): labels for training data
    - x_test (np.ndarray) (n, d): test data
    - k (int): number of neighbors to consider

    Returns:
    - predictions (np.ndarray) (n,): predicted labels for x_test
    
    '''
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if k > len(x_train):
        raise ValueError("k cannot be larger than the number of training samples")

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    predictions = []
    for test_point in x_test:
        distances = []
        # compute the distance between the test point and all training points
        for i, train_point in enumerate(x_train):
            # TODO: invoke the function euclidean_distance to compute the distance between two points
            distance = euclidean_distance(test_point, train_point)
            # store the distance
            distances.append((i, distance))
        
        # TODO: sort the distances 
        distances.sort(key=lambda item: item[1])

        # get the labels of the k nearest neighbor training samples
        top_k_labels = [y_train[distance[0]] for distance in distances[:k]]
        
        # select the most common class label among them
        label_counts = {}
        for label in top_k_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        prediction = max(label_counts, key=label_counts.get)
        predictions.append(prediction)

    return np.array(predictions)

def predict(x_train, y_train, x_test, y_test):
    # Set the number of neighbors and compute the accuracy of test data.
    y_pred = KNN_predict(x_train, y_train, x_test, K)
    acc = (np.sum(y_test == y_pred) / len(y_test)) * 100

    print("Test accuracy: {}".format(acc))
    return acc
