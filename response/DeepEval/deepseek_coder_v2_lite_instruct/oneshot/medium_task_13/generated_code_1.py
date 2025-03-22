import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def conv_block(input_tensor, filters):
        conv1 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        concatenated = Concatenate(axis=3)([input_tensor, conv3])
        return concatenated

    # First convolutional block
    x = conv_block(input_layer, filters=32)

    # Second convolutional block
    x = conv_block(x, filters=64)

    # Third convolutional block
    x = conv_block(x, filters=128)

    # Flatten the output
    flatten_layer = Flatten()(x)

    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model