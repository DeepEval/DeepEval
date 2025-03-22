import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, AveragePooling2D, Lambda, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split input channels and apply 1x1 convolutions
    split_channels = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=2))(input_layer)
    conv1 = Conv2D(filters=input_layer.shape[-1] // 3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_channels[0])
    conv2 = Conv2D(filters=input_layer.shape[-1] // 3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_channels[1])
    conv3 = Conv2D(filters=input_layer.shape[-1] // 3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_channels[2])

    # Average pooling
    pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
    pool3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)

    # Concatenate feature maps
    concatenated = Concatenate(axis=2)([pool1, pool2, pool3])

    # Flatten and fully connected layers
    flatten = Flatten()(concatenated)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model