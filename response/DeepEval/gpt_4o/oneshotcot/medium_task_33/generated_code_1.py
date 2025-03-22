import keras
from keras.layers import Input, Lambda, SeparableConv2D, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three channel groups
    def split_channels(input_tensor):
        return tf.split(input_tensor, num_or_size_splits=3, axis=-1)

    split_layer = Lambda(split_channels)(input_layer)

    # Apply separable convolutions to each channel group
    conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_layer[0])
    conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_layer[1])
    conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_layer[2])

    # Concatenate the outputs from the separable convolution layers
    concat_layer = Concatenate()([conv1, conv2, conv3])

    # Flatten and pass through dense layers for classification
    flatten_layer = Flatten()(concat_layer)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model