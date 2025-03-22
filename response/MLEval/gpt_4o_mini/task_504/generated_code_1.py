import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential

def method():
    # Example input data (sequence of integers)
    input_data = np.array([[1, 2, 3], [4, 5, 0]])  # Example with padding (0)

    # Define parameters for embedding
    vocab_size = 6   # Size of the vocabulary
    embedding_dim = 4  # Dimension of the embedding vector

    # Create an embedding layer
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_data.shape[1])

    # Create a simple model to use the embedding layer
    model = Sequential()
    model.add(embedding_layer)

    # Get the embedded output by passing the input data
    embedded_output = model(input_data)

    return embedded_output.numpy()  # Convert to numpy array for easier handling

# Call the method to validate
output = method()
print(output)