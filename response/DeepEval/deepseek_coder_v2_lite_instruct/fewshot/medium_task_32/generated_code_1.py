import keras
import tensorflow as tf
from keras.layers import Input, Lambda, Conv2D, DepthwiseConv2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the last dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))
    groups = split_layer(input_layer)

    # Feature extraction for each group using depthwise separable convolutions
    def feature_extraction(input_group, kernel_size):
        return DepthwiseConv2D(kernel_size=kernel_size, strides=(1, 1), padding='same', activation='relu')(input_group)

    conv1 = feature_extraction(groups[0], (1, 1))
    conv2 = feature_extraction(groups[1], (3, 3))
    conv3 = feature_extraction(groups[2], (5, 5))

    # Concatenate the outputs of the three groups
    concatenated = Concatenate()([conv1, conv2, conv3])

    # Flatten the concatenated features
    flattened = Flatten()(concatenated)

    # Pass through a fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model