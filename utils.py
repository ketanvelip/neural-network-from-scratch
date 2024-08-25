import pandas as pd
import numpy as np

def load_data(path):
    """
    Load dataset from a CSV file and return features and labels.
    """
    df = pd.read_csv(path)
    # Assuming the last column is the target label
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def normalize_features(X):
    """
    Normalize the features in X to have mean 0 and variance 1.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std
