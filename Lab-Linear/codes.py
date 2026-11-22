import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def closed_form_solution(X, y):
    '''
    Implement the closed form solution for linear regression

    Args:
        X (np.ndarray) (n, d): the input data
        y (np.ndarray) (n,): the ground truth label
    Returns:
        y_pred (np.ndarray) (n,): the predicted value
        theta (np.ndarray) (2,1): the weights
    '''
    # In this section, please implement the linear regression using the closed form solution 

    # TODO: firstly, add a column of 1s to the x_train as the bias term
    X = np.asarray(X)
    y = np.asarray(y)
    X_design = np.concatenate(
        (np.ones((X.shape[0], 1), dtype=X.dtype), X),
        axis=1,
    )

    # TODO: secondly, use the closed form solution to calculate the best theta
    theta = np.linalg.pinv(X_design) @ y.reshape(-1)

    # TODO: finally, compute the y_pred using the best theta
    y_pred = X_design @ theta

    return y_pred, theta

def predict(X, theta):
    ''' 
    Predict the output of the input data

    Args:
        X (np.ndarray) (n, d): the input data
        theta (np.ndarray) (2,1): the weights
    Returns:
        y_pred (np.ndarray) (n,): the predicted value
    '''
    # TODO: compute the y_pred using the input data and the weights
    X = np.asarray(X)
    theta = np.asarray(theta).reshape(-1)
    X_design = np.concatenate(
        (np.ones((X.shape[0], 1), dtype=X.dtype), X),
        axis=1,
    )
    y_pred = X_design @ theta

    return y_pred

# Define the loss function
def compute_loss(y_pred, label):
    ''' 
    Compute the loss function for linear regression

    Args:
        y_pred (np.ndarray) (n,): the predicted value
        label (np.ndarray) (n,): the ground truth label
    Returns:
        loss (float): the loss value
    '''
    # TODO: compute the loss using the y_pred and the label
    y_pred = np.asarray(y_pred)
    label = np.asarray(label)
    return float(np.mean((y_pred - label) ** 2))
