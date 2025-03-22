import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

def dl_model():
    input_layer = keras.Input(shape=(28, 28, 1))

    # First block
    avg_pool1 = layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    avg_pool1 = layers.Flatten()(avg_pool1)

    avg_pool2 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    avg_pool2 = layers.Flatten()(avg_pool2)

    avg_pool3 = layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    avg_pool3 = layers.Flatten()(avg_pool3)

    block_output = layers.Concatenate()([avg_pool1, avg_pool2, avg_pool3])

    # Dense layer
    dense_layer = layers.Dense(units=128, activation='relu')(block_output)
    dense_layer = layers.Reshape(target_shape=(1, 128))(dense_layer)

    # Second block
    def split(input_tensor):
        return tf.split(input_tensor, num_or_size_splits=4, axis=-1)

    split_layer = layers.Lambda(split)(dense_layer)

    conv1 = layers DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same')(split_layer[0])
    conv2 = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(split_layer[1])
    conv3 = layers.DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same')(split_layer[2])
    conv4 = layers.DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same')(split_layer[3])

    concat_output = layers.Concatenate()([conv1, conv2, conv3, conv4])

    flatten_layer = layers.Flatten()(concat_output)
    output_layer = layers.Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model