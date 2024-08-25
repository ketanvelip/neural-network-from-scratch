import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Load dataset from a CSV file and return features and labels.
    """
    df = pd.read_csv(filepath)
    X = df.iloc[:, :-1].values  # All columns except the last one are features
    y = df.iloc[:, -1].values   # The last column is the label
    return X, y

def normalize_features(X):
    """
    Normalize the features in X to have mean 0 and variance 1.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std
