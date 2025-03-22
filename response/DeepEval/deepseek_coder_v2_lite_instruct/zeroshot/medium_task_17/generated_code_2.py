import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Permute, Reshape, Dense

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)
    
    # Define the input layer
    inputs = Input(shape=input_shape)
    
    # Reshape the input tensor into (height, width, groups, channels_per_group)
    reshaped = Reshape(target_shape=(32, 32, 3, 1))(inputs)
    
    # Permute the dimensions to swap the third and fourth dimensions
    permuted = Permute((1, 2, 4, 3))(reshaped)
    
    # Reshape back to the original input shape
    final_shape = (32, 32, 3)
    reshaped_back = Reshape(target_shape=final_shape)(permuted)
    
    # Flatten the reshaped tensor
    flattened = tf.keras.layers.Flatten()(reshaped_back)
    
    # Pass through a fully connected layer with softmax activation
    outputs = Dense(units=10, activation='softmax')(flattened)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()