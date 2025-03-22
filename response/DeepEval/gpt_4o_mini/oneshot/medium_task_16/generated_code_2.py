import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into 3 groups along the channel dimension
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Process each group with 1x1 convolution
    conv_outputs = []
    for group in split_inputs:
        # Number of filters set to one-third of the input channels
        num_filters = 1
        conv = Conv2D(filters=num_filters, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(group)
        # Downsampling with average pooling
        pooled = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)
        conv_outputs.append(pooled)

    # Concatenate the outputs along the channel dimension
    concatenated = Concatenate(axis=-1)(conv_outputs)

    # Flatten the result
    flatten_layer = Flatten()(concatenated)

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model