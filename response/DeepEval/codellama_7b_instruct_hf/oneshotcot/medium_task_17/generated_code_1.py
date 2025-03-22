import keras
from keras.layers import Input, Reshape, Permute, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Reshape input tensor into groups of 3 channels
    reshaped_input = Reshape((32, 32, 3, 1))(input_layer)
    
    # Permute third and fourth dimensions to enable channel shuffling
    shuffled_input = Permute((0, 1, 3, 2))(reshaped_input)
    
    # Reshape back to original input shape
    reshaped_shuffled_input = Reshape((32, 32, 1, 3))(shuffled_input)
    
    # Pass through fully connected layer with softmax activation
    output_layer = Dense(10, activation='softmax')(reshaped_shuffled_input)
    
    # Create and return model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model