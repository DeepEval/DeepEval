import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def conv_block(input_tensor, filters):
        conv1 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)
        return pool

    # First convolutional block
    conv1_output = conv_block(input_layer, filters=32)

    # Second convolutional block
    conv2_output = conv_block(conv1_output, filters=64)

    # Third convolutional block
    conv3_output = conv_block(conv2_output, filters=128)

    # Concatenate outputs along the channel dimension
    concatenated = Concatenate(axis=-1)([input_layer, conv1_output, conv2_output, conv3_output])

    # Batch normalization
    batch_norm = BatchNormalization()(concatenated)

    # Flatten the result
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model