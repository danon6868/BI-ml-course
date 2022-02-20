import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    TP = FP = TN = FN = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            if y_true[i] == 1:
                TP += 1
            else:
                TN += 1
        else:
            if y_true[i] == 1:
                FP += 1
            else:
                FN += 1

    precision = TP / (TP + FP) if TP + FP != 0 else None
    recall = TP / (TP + FN) if TP + FN != 0 else None
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else None
    f1 = ((2 * precision * recall) / (precision + recall) 
          if (precision is not None) and (recall is not None) and (precision + recall) != 0 
          else None)
    return precision, recall, accuracy, f1
           


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    """
    YOUR CODE IS HERE
    """
    pass


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    sum1 = sum2 = 0
    mean_y = np.mean(y_true)
    for i in range(len(y_true)):
        sum1 += (y_true[i] - y_pred[i]) **2
        sum2 += (y_true[i] - mean_y) ** 2
    r2 = 1 - (sum1 / sum2)
    return r2

    


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """
    sum = 0
    N = len(y_true)
    for i in range(N):
        sum += (y_true[i] - y_pred[i]) ** 2
    mse = (1 / N) * sum
    return mse



def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """
    sum = 0
    N = len(y_true)
    for i in range(N):
        sum += abs(y_true[i] - y_pred[i])
    mae = (1 / N) * sum
    return mae
