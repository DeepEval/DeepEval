import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_1 = np.random.randn(input_size, hidden_size)
        self.weights_2 = np.random.randn(hidden_size, output_size)
        self.bias_1 = np.zeros(hidden_size)
        self.bias_2 = np.zeros(output_size)

    def forward(self, X):
        hidden = np.dot(X, self.weights_1) + self.bias_1
        hidden = np.maximum(hidden, 0)
        output = np.dot(hidden, self.weights_2) + self.bias_2
        return output

if __name__ == "__main__":
    # Example usage
    model = MLP(2, 4, 1)
    input_data = np.array([[0.1, 0.2]])
    output = model.forward(input_data)
    print(output)