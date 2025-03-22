import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape, Permute, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Reshape the input tensor into (height, width, groups, channels_per_group)
    reshaped_layer = Reshape((32, 32, 3, 1))(input_layer)
    
    # Swap the third and fourth dimensions
    permuted_layer = Permute((1, 2, 4, 3))(reshaped_layer)
    
    # Reshape back to the original input shape
    reshaped_back_layer = Reshape((32, 32, 3))(permuted_layer)
    
    # Flatten the reshaped tensor
    flattened_layer = Flatten()(reshaped_back_layer)
    
    # Fully connected layer with softmax activation for classification
    output_layer = Dense(10, activation='softmax')(flattened_layer)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()