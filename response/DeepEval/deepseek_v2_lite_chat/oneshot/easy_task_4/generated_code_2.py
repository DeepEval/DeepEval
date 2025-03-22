import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, BatchNormalization

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # First block
    def first_block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
        return pool1

    first_block_output = first_block(input_tensor=input_layer)

    # Second block
    def second_block(input_tensor):
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv4)
        return pool2

    second_block_output = second_block(input_tensor=first_block_output)

    # Flatten and fully connected layers
    flatten = Flatten()(second_block_output)
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model