import tensorflow as tf

def method():
    # Sample input data (replace with your actual input)
    input_data = ["This", "is", "a", "sample", "sentence"]
    
    # Define the embedding layer
    embedding_layer = tf.keras.layers.Embedding(input_dim=10000,  # Vocabulary size (adjust as needed)
                                                output_dim=128)  # Embedding dimension
    
    # Apply the embedding layer to the input data
    embedded_sequence = embedding_layer(input_data)

    return embedded_sequence

# Call the method and print the output
output = method()
print(output)