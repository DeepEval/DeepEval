import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Permute, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Group the channels into 3 groups
    reshaped = Reshape(target_shape=(32, 32, 3, 1))(input_layer)
    
    # Swap the third and fourth dimensions
    permuted = Permute((1, 2, 4, 3))(reshaped)
    
    # Reshape back to the original shape
    reshaped_back = Reshape(target_shape=(32, 32, 3))(permuted)
    
    # Flatten the tensor
    flattened = Flatten()(reshaped_back)
    
    # Pass through a fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flattened)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model