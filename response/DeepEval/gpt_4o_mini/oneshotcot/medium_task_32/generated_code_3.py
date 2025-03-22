import keras
import tensorflow as tf
from keras.layers import Input, Lambda, DepthwiseConv2D, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the last dimension
    split_input = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Apply depthwise separable convolutions with different kernel sizes
    path1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(split_input[0])
    path2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(split_input[1])
    path3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(split_input[2])

    # Concatenate the outputs of the three paths
    concatenated = Concatenate()([path1, path2, path3])

    # Flatten the concatenated output
    flattened = Flatten()(concatenated)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model