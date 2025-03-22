import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Split the input into three channels
    split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Define feature extraction with separable convolutions for each channel
    def separable_conv_block(channel_input):
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(channel_input)
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(channel_input)
        conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(channel_input)
        return conv1x1, conv3x3, conv5x5

    # Apply the separable convolution block to each channel
    outputs = [separable_conv_block(channel) for channel in split_channels]

    # Concatenate all outputs from the separable convolutions
    conv1x1_outputs = Concatenate()(outputs[0])  # 1x1 outputs from each channel
    conv3x3_outputs = Concatenate()(outputs[1])  # 3x3 outputs from each channel
    conv5x5_outputs = Concatenate()(outputs[2])  # 5x5 outputs from each channel
    concatenated_outputs = Concatenate()([conv1x1_outputs, conv3x3_outputs, conv5x5_outputs])

    # Flatten the concatenated outputs
    flatten_layer = Flatten()(concatenated_outputs)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model