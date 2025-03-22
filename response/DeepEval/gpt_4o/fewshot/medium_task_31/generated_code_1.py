import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Flatten, Dense, Concatenate, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    input_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Apply different convolutional kernels to each group
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_groups[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_groups[1])
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_groups[2])

    # Concatenate the outputs from the three groups
    concatenated = Concatenate()([conv1, conv2, conv3])

    # Flatten the concatenated features
    flatten = Flatten()(concatenated)

    # Pass through two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model