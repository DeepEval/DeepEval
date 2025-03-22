import keras
from keras.layers import Input, AveragePooling2D, Flatten, Dense, concatenate, Reshape, Lambda, DepthwiseConv2D, Conv2D
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # First block
    pool_1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    pool_2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    pool_3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)
    pool_1 = Flatten()(pool_1)
    pool_2 = Flatten()(pool_2)
    pool_3 = Flatten()(pool_3)
    concat_pool = concatenate([pool_1, pool_2, pool_3])

    # Reshape and fully connected layer
    reshape_layer = Reshape((4, 1))(concat_pool)
    dense_1 = Dense(units=10, activation='relu')(reshape_layer)

    # Second block
    split_layer = Lambda(tf.split)(dense_1, num_or_size_splits=4, axis=-1)
    conv_1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same')(split_layer[0])
    conv_2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(split_layer[1])
    conv_3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same')(split_layer[2])
    conv_4 = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same')(split_layer[3])
    concat_conv = concatenate([conv_1, conv_2, conv_3, conv_4])

    # Flatten and output layer
    flatten_layer = Flatten()(concat_conv)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model