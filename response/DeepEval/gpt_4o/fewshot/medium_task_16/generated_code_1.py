import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split input into three groups along the channel dimension
    def split_input(input_tensor):
        return tf.split(input_tensor, num_or_size_splits=3, axis=-1)

    split_layer = Lambda(split_input)(input_layer)

    # Apply 1x1 convolutions to each group independently
    conv1 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    conv2 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
    conv3 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[2])

    # Downsample each group via an average pooling layer
    pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
    pool3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)

    # Concatenate the feature maps along the channel dimension
    concatenated = Concatenate(axis=-1)([pool1, pool2, pool3])

    # Flatten the feature maps into a one-dimensional vector
    flatten_layer = Flatten()(concatenated)

    # Pass through two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model