import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Splitting the input along the channel dimension into three groups
    def split_channels(x):
        return tf.split(x, num_or_size_splits=3, axis=-1)

    split_layer = Lambda(split_channels)(input_layer)

    # Applying 1x1 convolutions to each group independently
    conv1 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    conv2 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
    conv3 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[2])

    # Downsampling using average pooling
    pool1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
    pool3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)

    # Concatenating the resulting groups along the channel dimension
    concatenated = Concatenate(axis=-1)([pool1, pool2, pool3])

    # Flattening and adding fully connected layers
    flatten_layer = Flatten()(concatenated)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Building the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model