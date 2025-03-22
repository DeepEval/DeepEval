import keras
import tensorflow as tf
from keras.layers import Input, Reshape, Permute, Dense, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels

    # Reshape the input tensor to (height, width, groups, channels_per_group)
    # where groups = 3 and channels_per_group = 1 (since 3 channels total)
    reshaped = Reshape((32, 32, 3, 1))(input_layer)  # Now shape is (32, 32, 3, 1)

    # Permute the dimensions to swap the last two dimensions (channels and channels_per_group)
    shuffled = Permute((0, 1, 3, 2))(reshaped)  # Now shape is (32, 32, 1, 3)

    # Reshape back to the original input shape
    reshaped_back = Reshape((32, 32, 3))(shuffled)  # Back to (32, 32, 3)

    # Flatten the output for the fully connected layer
    flatten_layer = Flatten()(reshaped_back)

    # Fully connected layer for classification with softmax activation
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model