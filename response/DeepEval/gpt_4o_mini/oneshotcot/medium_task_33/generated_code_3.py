import keras
from keras.layers import Input, Conv2D, Concatenate, Lambda, Flatten, Dense
import tensorflow as tf

def dl_model():
    # Define input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels

    # Split the input into three channel groups
    def split_channels(x):
        return tf.split(x, num_or_size_splits=3, axis=-1)

    channel_groups = Lambda(split_channels)(input_layer)

    # Define convolutional paths for each channel group
    def create_conv_path(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(input_tensor)
        return conv1, conv2, conv3
    
    # Process each channel group
    conv_outputs = []
    for group in channel_groups:
        conv_outputs.append(create_conv_path(group))

    # Concatenate all the outputs from the different paths
    concatenated = Concatenate()([conv_output for group_outputs in conv_outputs for conv_output in group_outputs])

    # Flatten the concatenated output
    flatten_layer = Flatten()(concatenated)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # 10 classes for CIFAR-10

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model