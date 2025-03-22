import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Permute, Flatten, Dense, Softmax
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_tensor = Input(shape=(32, 32, 3)) 

    # Reshape for channel shuffling
    input_shape = tf.shape(input_tensor)
    reshaped_input = Reshape((input_shape[1], input_shape[2], 3, input_shape[3] // 3))(input_tensor)
    
    # Channel shuffling
    shuffled_input = Permute((2, 3, 1, 4))(reshaped_input)

    # Reshape back to original shape
    reshaped_output = Reshape((input_shape[1], input_shape[2], input_shape[3]))(shuffled_input)

    # Flatten and fully connected layer
    flattened = Flatten()(reshaped_output)
    output = Dense(10, activation='softmax')(flattened)  

    # Create the model
    model = Model(inputs=input_tensor, outputs=output)
    return model