import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Define the convolutional blocks for each group
    def conv_block(input_tensor, filters, kernel_size):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu')(input_tensor)
        return conv

    # Apply convolutional layers to each group
    conv1x1 = conv_block(split_layer[0], filters=32, kernel_size=(1, 1))
    conv3x3 = conv_block(split_layer[1], filters=32, kernel_size=(3, 3))
    conv5x5 = conv_block(split_layer[2], filters=32, kernel_size=(5, 5))

    # Concatenate the outputs of the three groups
    concatenated = Concatenate(axis=-1)([conv1x1, conv3x3, conv5x5])

    # Batch normalization and flattening
    batch_norm = BatchNormalization()(concatenated)
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model