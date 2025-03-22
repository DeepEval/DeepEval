import keras
from keras.layers import Input, Conv2D, Multiply, Softmax, Add, GlobalAveragePooling2D, Dense, ReLU, LayerNormalization
from keras.layers import Conv2D as Conv, Lambda, Concatenate, Flatten
import tensorflow as tf

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Generate attention weights with a 1x1 convolution followed by a softmax layer
    weights = Conv(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=None)(input_layer)
    weights = Softmax()(weights)

    # Multiply the input features with the attention weights to obtain contextual information
    contextual = Multiply()([input_layer, weights])

    # Reduce the input dimensionality to one-third of its original size using another 1x1 convolution
    reduce_dim = Conv(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(contextual)

    # Apply layer normalization and ReLU activation
    normalize = LayerNormalization()(reduce_dim)
    normalize = ReLU()(normalize)

    # Restore the dimensionality with an additional 1x1 convolution
    restore_dim = Conv(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=None)(normalize)

    # Add the processed output to the original input image
    add_layer = Add()([restore_dim, input_layer])

    # Flatten the output
    flatten_layer = Flatten()(add_layer)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model