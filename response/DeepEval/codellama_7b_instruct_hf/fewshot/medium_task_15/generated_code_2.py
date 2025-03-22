import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Flatten, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    # Global average pooling
    pool = MaxPooling2D(pool_size=(8, 8), strides=(8, 8), padding='same')(conv)
    pool = Flatten()(pool)

    # Fully connected layers
    dense1 = Dense(units=64, activation='relu')(pool)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Reshape output
    output_layer = Reshape(target_shape=(1, 1, 10))(dense2)

    # Weighted feature maps
    input_weighted = input_layer * output_layer
    input_weighted = Flatten()(input_weighted)

    # Concatenate with input
    concat_layer = Concatenate()([input_weighted, dense2])

    # 1x1 convolution and average pooling
    conv = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same')(concat_layer)
    pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv)

    # Final fully connected layer
    flatten = Flatten()(pool)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model