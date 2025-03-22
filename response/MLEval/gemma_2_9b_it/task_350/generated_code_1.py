import tensorflow as tf

def method():
    # Define the input data
    input_data = tf.constant(["This is an example sentence."]) 

    # Define the embedding layer
    embedding_layer = tf.keras.layers.Embedding(input_dim=10000, output_dim=128) 

    # Apply the embedding layer to the input data
    output = embedding_layer(input_data) 

    return output

# Call the method and print the output
output = method()
print(output)