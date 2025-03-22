import keras
from keras.layers import Input, Dense, Reshape, Permute, Flatten
from keras.models import Model

def dl_model():
    # CIFAR-10 images have the shape (32, 32, 3)
    input_layer = Input(shape=(32, 32, 3))
    
    # Reshape to (height, width, groups, channels_per_group)
    groups = 3
    channels_per_group = 3 // groups
    reshaped = Reshape(target_shape=(32, 32, groups, channels_per_group))(input_layer)
    
    # Swap the third and fourth dimensions for channel shuffling
    shuffled = Permute((1, 2, 4, 3))(reshaped)
    
    # Reshape back to the original input shape
    reshaped_back = Reshape(target_shape=(32, 32, 3))(shuffled)
    
    # Flatten the output before the fully connected layer
    flattened = Flatten()(reshaped_back)
    
    # Fully connected layer with softmax activation for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)
    
    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model