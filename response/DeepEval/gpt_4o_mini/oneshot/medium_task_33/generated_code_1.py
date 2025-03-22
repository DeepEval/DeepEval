import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Dense, Flatten
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input image into three channel groups
    split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    def separable_conv_block(channel_input):
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu', use_bias=False)(channel_input)
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', use_bias=False)(channel_input)
        conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', use_bias=False)(channel_input)
        return conv1x1, conv3x3, conv5x5

    # Apply separable convolution to each channel group
    outputs = [separable_conv_block(channel) for channel in split_channels]
    
    # Flatten the output of each block and concatenate
    conv1x1_outputs = [output[0] for output in outputs]
    conv3x3_outputs = [output[1] for output in outputs]
    conv5x5_outputs = [output[2] for output in outputs]

    concatenated = Concatenate()([*conv1x1_outputs, *conv3x3_outputs, *conv5x5_outputs])

    # Flatten the concatenated outputs
    flatten_layer = Flatten()(concatenated)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model