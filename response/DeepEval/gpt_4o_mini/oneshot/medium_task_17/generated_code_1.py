import keras
from keras.layers import Input, Reshape, Permute, Dense, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 input shape

    # Reshape the input tensor
    reshaped = Reshape((32, 32, 3, 1))(input_layer)  # Shape to (height, width, groups, channels_per_group)
    
    # Permute the dimensions to shuffle channels
    permuted = Permute((1, 2, 4, 3))(reshaped)  # Shape to (height, width, channels_per_group, groups)
    
    # Reshape back to original input shape
    reshaped_back = Reshape((32, 32, 3))(permuted)  # Back to original shape (height, width, channels)

    # Flatten the output for the Dense layer
    flatten_layer = Flatten()(reshaped_back)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # 10 classes for CIFAR-10

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model