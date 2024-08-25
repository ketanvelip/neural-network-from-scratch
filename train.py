import numpy as np
from utils import load_data, normalize_features
from neural_network import NeuralNetwork

def one_hot_encode(y):
    """
    One-hot encode the labels for classification.
    """
    n_classes = np.max(y) + 1
    return np.eye(n_classes)[y]

def main():
    # Load and preprocess data
    X, y = load_data('data/dataset.csv')
    X = normalize_features(X)
    y = one_hot_encode(y)  # Assume classification problem

    # Define the neural network
    input_size = X.shape[1]
    hidden_size = 10
    output_size = y.shape[1]
    learning_rate = 0.01

    nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
    
    # Train the model
    nn.train(X, y, epochs=1000)
    
    # Test the model
    predictions = nn.predict(X)
    accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
    print(f'Accuracy: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    main()
