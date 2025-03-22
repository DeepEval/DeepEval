import keras
import tensorflow as tf
from keras.layers import Input, Lambda, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    def split_channels(input_tensor):
        return tf.split(input_tensor, num_or_size_splits=3, axis=-1)

    # Lambda layer to perform channel splitting
    split_layers = Lambda(split_channels)(input_layer)

    # Apply 1x1 convolutions to each group independently
    conv_layers = []
    for split_layer in split_layers:
        conv_layer = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer)
        conv_layers.append(conv_layer)

    # Downsample each group using average pooling
    pooled_layers = []
    for conv_layer in conv_layers:
        pooled_layer = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_layer)
        pooled_layers.append(pooled_layer)

    # Concatenate the pooled groups along the channel dimension
    concatenated_layer = Concatenate(axis=-1)(pooled_layers)

    # Flatten and pass through two fully connected layers for classification
    flatten_layer = Flatten()(concatenated_layer)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model