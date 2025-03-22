import keras
from keras.layers import Input, Lambda, DepthwiseConv2D, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the last dimension
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Apply depthwise separable convolutional layers to each group
    depthwise_1x1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(split_inputs[0])
    depthwise_3x3 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(split_inputs[1])
    depthwise_5x5 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(split_inputs[2])

    # Concatenate the outputs of the three paths
    concatenated = Concatenate()([depthwise_1x1, depthwise_3x3, depthwise_5x5])

    # Flatten the concatenated output
    flatten_layer = Flatten()(concatenated)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model