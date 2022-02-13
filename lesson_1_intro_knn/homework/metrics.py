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

    true_positive  = tp = np.sum((y_pred == 1) * (y_true == 1))
    false_positive = fp = np.sum((y_pred == 1) * (y_true == 0))
    true_negative  = tn = np.sum((y_pred == 0) * (y_true == 0))
    false_negative = fn = np.sum((y_pred == 0) * (y_true == 1))

    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0.0
        
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0.0
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    try:
        f1_score = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        f1_score = 0.0
    

    return precision, recall, f1_score, accuracy


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    accuracy = np.sum(y_pred == y_true) / y_true.shape[0]
    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    r2_score = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    return r2_score


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    mse = (1 / y_true.shape[0]) * np.sum((y_true - y_pred) ** 2)
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

    mae = (1 / y_true.shape[0]) * np.sum(np.abs(y_true - y_pred))
    return mae
    