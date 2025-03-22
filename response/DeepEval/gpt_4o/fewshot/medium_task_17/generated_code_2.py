import keras
import tensorflow as tf
from keras.layers import Input, Reshape, Permute, Flatten, Dense

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 input shape

    input_layer = Input(shape=input_shape)
    
    # Step 1: Reshape into (height, width, groups, channels_per_group)
    # For CIFAR-10 with 3 channels, groups=3, and each group will have 1 channel
    reshaped = Reshape(target_shape=(32, 32, 3, 1))(input_layer)

    # Step 2: Permute to swap the third and fourth dimensions (channel shuffle)
    shuffled = Permute((1, 2, 4, 3))(reshaped)

    # Step 3: Reshape back to the original shape
    reshaped_back = Reshape(target_shape=input_shape)(shuffled)

    # Step 4: Flatten and pass through fully connected layer for classification
    flatten_layer = Flatten()(reshaped_back)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model