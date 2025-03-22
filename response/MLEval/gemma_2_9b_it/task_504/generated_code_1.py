import tensorflow as tf

def method():
    # Define the input data
    input_data = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]])  

    # Create a simple embedding layer
    embedding_layer = tf.keras.layers.Embedding(input_dim=10, output_dim=8)  

    # Apply the embedding layer to the input data
    output = embedding_layer(input_data)

    return output

# Call the method and print the output
output = method()
print(output)