import keras
from keras.layers import Input, Dense, Reshape, Permute, Flatten

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 with 3 color channels
    input_layer = Input(shape=input_shape)
    
    # Reshape input into shape (height, width, groups, channels_per_group)
    groups = 3
    channels_per_group = input_shape[2] // groups
    reshaped = Reshape((input_shape[0], input_shape[1], groups, channels_per_group))(input_layer)
    
    # Permute the dimensions to enable channel shuffling
    permuted = Permute((1, 2, 4, 3))(reshaped)
    
    # Reshape back to original shape
    reshaped_back = Reshape(input_shape)(permuted)
    
    # Flatten the tensor before passing to the fully connected layer
    flattened = Flatten()(reshaped_back)
    
    # Fully connected layer with softmax activation for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model