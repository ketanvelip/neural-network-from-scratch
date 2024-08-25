import numpy as np
from utils import load_data, normalize_features
from neural_network import NeuralNetwork

def one_hot_encode(labels):
    """
    One-hot encode the labels for classification.
    """
    n_classes = np.max(labels) + 1
    return np.eye(n_classes)[labels]

def main():
    # Load and preprocess the training data
    X_train, y_train = load_data('data/dataset.csv')
    X_train = normalize_features(X_train)
    y_train = one_hot_encode(y_train)  # Assuming a classification problem

    # Define the neural network
    input_size = X_train.shape[1]
    hidden_size = 10
    output_size = y_train.shape[1]
    learning_rate = 0.01

    nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
    
    # Train the model
    nn.train(X_train, y_train, epochs=1000)
    
    # Evaluate the model on the training data
    train_predictions = nn.predict(X_train)
    train_accuracy = np.mean(np.argmax(train_predictions, axis=1) == np.argmax(y_train, axis=1))
    print(f'Training Accuracy: {train_accuracy * 100:.2f}%')

    # Load and preprocess the test data
    X_test, y_test = load_data('data/test_data.csv')  # Path to the test data CSV file
    X_test = normalize_features(X_test)
    y_test = one_hot_encode(y_test)  # Assuming a classification problem

    # Predict the labels for the test data
    test_predictions = nn.predict(X_test)
    
    # Calculate and print test accuracy
    test_accuracy = np.mean(np.argmax(test_predictions, axis=1) == np.argmax(y_test, axis=1))
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

if __name__ == "__main__":
    main()
