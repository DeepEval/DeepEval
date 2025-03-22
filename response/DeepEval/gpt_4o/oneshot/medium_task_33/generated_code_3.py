import keras
from keras.layers import Input, Lambda, SeparableConv2D, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    # Define input layer for CIFAR-10 images
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three separate channels
    split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Define separable convolutions for each split channel
    conv_layers = []
    for i in range(3):
        sep_conv_1x1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_channels[i])
        sep_conv_3x3 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_channels[i])
        sep_conv_5x5 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_channels[i])

        # Concatenate the separable convolutions
        concatenated = Concatenate()([sep_conv_1x1, sep_conv_3x3, sep_conv_5x5])
        conv_layers.append(concatenated)

    # Concatenate the results from all channel groups
    concatenated_channels = Concatenate()(conv_layers)

    # Flatten the concatenated outputs
    flatten_layer = Flatten()(concatenated_channels)

    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model