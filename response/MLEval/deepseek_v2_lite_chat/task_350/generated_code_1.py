import tensorflow as tf

def method(input_data, embedding_dim=128, input_length=10):
    # Define the embedding layer
    embedding = tf.keras.layers.Embedding(input_dim=10000, output_dim=embedding_dim, input_length=input_length)
    
    # Input data shape should be (None, input_length)
    # For example: input_data = [[1, 2, 3], [4, 5, 6], ...]
    
    # Fit the embedding matrix
    embedding.build(tf.TensorShape([input_data.shape[0], input_data.shape[1]]))
    
    # Get the embeddings
    embedded_seq = embedding(input_data)
    
    return embedded_seq

# Example usage
input_data = tf.constant([[5000, 3, 4], [1000, 4, 5], [2000, 5, 6], [8000, 6, 7]])
output = method(input_data)

# Print the output to verify
print("Output:", output)