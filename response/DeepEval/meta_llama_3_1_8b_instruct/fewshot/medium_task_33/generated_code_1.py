import keras
from keras.layers import Input, Lambda, DepthwiseConv2D, Concatenate, Dense
import tensorflow as tf

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three channel groups using Lambda layer
    def split_channels(input_tensor):
        return Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)

    split_output = split_channels(input_layer)

    # Feature extraction for each group using separable convolutional layers
    conv1_group1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_output[0])
    conv1_group2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_output[1])
    conv1_group3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_output[2])

    # Concatenate the outputs from the three groups
    concat_output = Concatenate()([conv1_group1, conv1_group2, conv1_group3])

    # Flatten the concatenated output
    flatten_output = Flatten()(concat_output)

    # Pass the output through three fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model