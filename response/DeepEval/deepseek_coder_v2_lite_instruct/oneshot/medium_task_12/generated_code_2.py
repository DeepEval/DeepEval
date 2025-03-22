import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor, filters):
        conv = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv = BatchNormalization()(conv)
        return conv

    # First block
    conv1 = block(input_layer, filters=32)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv1)

    # Second block
    conv2 = block(pool1, filters=64)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)

    # Third block
    conv3 = block(pool2, filters=128)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv3)

    # Concatenate outputs of the blocks
    concatenated = Concatenate(axis=-1)([conv1, conv2, conv3])

    # Flatten the result
    flatten_layer = Flatten()(concatenated)

    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model